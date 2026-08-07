import CryptoKit
import Foundation

/// Compiled assertions for PushCrypto (docs/push-architecture.md §2) — the
/// same "compiled checks file" pattern the timeline math used. NOT part of
/// any app target (project.yml only globs Vigilume/Sources,
/// NotificationService/Sources and Shared). Run on macOS:
///
///     cd ios && swiftc -parse-as-library \
///         Shared/PushCrypto.swift Scripts/PushCryptoCheck.swift \
///         -o /tmp/push-crypto-check && /tmp/push-crypto-check
///
/// The known-answer vector below was generated with an INDEPENDENT AES-GCM
/// implementation (Python `cryptography`'s AESGCM — the same library shape
/// the hoster backend will use), laid out per the pinned wire format
/// base64(nonce12 || ciphertext || tag16):
///
///     key   = bytes(range(32))              # 000102...1f
///     nonce = bytes(range(0xA0, 0xAC))      # a0a1...ab
///     plaintext = {"title":"Person detected at Front Yard","body":"2 in
///       frame","event_id":"123","snapshot_url":"https://nvr.example.com/api/
///       events/123/snapshot.jpg?token=abc"}   (compact JSON, UTF-8)
///     combined = nonce + AESGCM(key).encrypt(nonce, plaintext, None)
@main
enum PushCryptoCheck {
    static var failures = 0

    static func check(_ ok: Bool, _ name: String) {
        if ok {
            print("PASS  \(name)")
        } else {
            print("FAIL  \(name)")
            failures += 1
        }
    }

    static func main() {
        // MARK: Known-answer vector (cross-implementation)

        let keyHex = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
        let encB64 = """
        oKGio6SlpqeoqaqrnToIRDGnZ51YR9e2dQmvsFDIPGT31DYJ+C5H8l/tB268AmemzlA3H3O+Zqdt\
        A6HDZSlmIQzwfAwgM258iFLgxtHS8TBZwcba1tPNK63i7smjatvXra7tJZsHUriLuCXg+/RYkSyB\
        4e3s9WMo5OWdREF+AsGzGvpMRuC70tE7hj7qIIJM/DHudkcqluRmZPum7+bZBNW1lLGcr97X0g2d\
        Ibmd80Z4fZTGsZ1lTlE=
        """
        let keyBytes = stride(from: 0, to: keyHex.count, by: 2).map { i -> UInt8 in
            let start = keyHex.index(keyHex.startIndex, offsetBy: i)
            let end = keyHex.index(start, offsetBy: 2)
            return UInt8(keyHex[start ..< end], radix: 16)!
        }
        let key = SymmetricKey(data: Data(keyBytes))

        let expected = PushCrypto.Payload(
            title: "Person detected at Front Yard",
            body: "2 in frame",
            eventID: "123",
            snapshotURL: "https://nvr.example.com/api/events/123/snapshot.jpg?token=abc"
        )

        let decrypted = try? PushCrypto.decrypt(encB64, using: key)
        check(decrypted == expected, "known-answer vector decrypts to the contract JSON")

        // MARK: Wrong key must be rejected (GCM auth tag)

        let wrongKey = SymmetricKey(size: .bits256)
        check(
            (try? PushCrypto.decrypt(encB64, using: wrongKey)) == nil,
            "wrong key is rejected"
        )

        // MARK: Tampered ciphertext must be rejected

        // (The line continuations in the literal above leave no newlines.)
        if var tampered = Data(base64Encoded: encB64) {
            tampered[20] ^= 0xFF
            check(
                (try? PushCrypto.decrypt(tampered.base64EncodedString(), using: key)) == nil,
                "tampered ciphertext is rejected"
            )
        } else {
            check(false, "vector literal is valid base64")
        }

        // MARK: Round trip (seal mirrors the hoster's encrypt side)

        let payload = PushCrypto.Payload(
            title: "Doorbell pressed", body: "Front Door",
            eventID: "9876", snapshotURL: nil
        )
        let roundTrip = (try? PushCrypto.seal(payload, using: key))
            .flatMap { try? PushCrypto.decrypt($0, using: key) }
        check(roundTrip == payload, "seal -> decrypt round trip (snapshot_url null)")

        // MARK: Per-camera grouping fields survive seal -> decrypt

        let grouped = PushCrypto.Payload(
            title: "Person detected at Backyard", body: "1 in frame",
            eventID: "99", snapshotURL: nil,
            camera: "backyard", cameraLabel: "Backyard"
        )
        let groupedRT = (try? PushCrypto.seal(grouped, using: key))
            .flatMap { try? PushCrypto.decrypt($0, using: key) }
        check(groupedRT == grouped, "camera + camera_label survive seal -> decrypt")
        check(groupedRT?.camera == "backyard" && groupedRT?.cameraLabel == "Backyard",
              "grouping fields decode to the thread id + summary")
        // Absent grouping fields decode to nil (older-server back-compat).
        check(roundTrip?.camera == nil && roundTrip?.cameraLabel == nil,
              "omitted camera/camera_label decode to nil")

        // MARK: Garbage inputs never throw out of decryptWithAnyStoredKey's shape

        check((try? PushCrypto.decrypt("not base64!!!", using: key)) == nil, "bad base64 rejected")
        check((try? PushCrypto.decrypt("AAAA", using: key)) == nil, "too-short input rejected")

        if failures > 0 {
            print("\(failures) check(s) FAILED")
            exit(1)
        }
        print("All PushCrypto checks passed")
    }
}
