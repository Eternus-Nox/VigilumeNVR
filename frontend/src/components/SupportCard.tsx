/**
 * Settings → System → "Support Vigilume": optional one-time or monthly support
 * for the project, via Stripe.
 *
 * WHY PAYMENT LINKS AND NOT THE STRIPE API. Vigilume is self-hosted: this code
 * runs on other people's boxes. Every server-side Stripe integration (Checkout
 * Sessions, Billing, webhooks) needs a `sk_live_…` SECRET key, and there is no
 * version of shipping the project's secret key to strangers that is safe — the
 * first person to `docker exec` into their own backend owns the Stripe account.
 * A Payment Link is just a URL. It is public by construction, so there is
 * nothing to leak, no key to rotate, no webhook endpoint to authenticate, and
 * no PCI scope anywhere in this repo. Card data never touches Vigilume; Stripe
 * hosts the whole checkout. Do not "upgrade" this to the API.
 *
 * WHY WEB-ONLY. The iOS app deliberately says nothing about this. App Review
 * guideline 3.1.3(f) forbids free companion apps from carrying "calls to action
 * for purchase outside of the app", which a donate button plainly is. Apple's
 * 3.1.1(a) update (post-Epic, April 2025) does now permit external purchase
 * links in US-storefront apps without an entitlement, so this is *allowed* — but
 * reviewers are inconsistent and it is not worth a rejection on the app's first
 * submission. Revisit once the app is approved; if it ever lands in the iOS app,
 * expect Apple to take a cut (Ninth Circuit, Dec 2025).
 *
 * Self-hosted UX: this card is admin-only (the whole Settings router is), so a
 * viewer never sees a solicitation. It renders NOTHING when no link is
 * configured below, which is what keeps a fork or a half-finished setup from
 * shipping a dead "Donate" button.
 */

/**
 * Stripe Payment Links — create at Dashboard → Payment Links (test mode first).
 *
 * One-time: a "Customers choose what to pay" link. Monthly: a recurring price.
 * Both must be LIVE-mode links (`https://buy.stripe.com/…`); a `test_` link
 * takes real-looking input and charges nothing, which is worse than no button.
 *
 * Empty string = that button is hidden. Both empty = the whole card is hidden.
 */
const ONE_TIME_URL = '';
const MONTHLY_URL = '';

export default function SupportCard() {
  if (!ONE_TIME_URL && !MONTHLY_URL) return null;

  return (
    <section className="card">
      <h2>Support Vigilume</h2>
      <p className="muted small">
        Vigilume is free, self-hosted, and has no account, no cloud, and nothing
        to upsell — your footage never leaves this box. If it&rsquo;s useful to
        you, you can chip in toward its future. Entirely optional; nothing in the
        app is gated behind it, and nothing changes if you don&rsquo;t.
      </p>
      <div className="row-inline wrap">
        {ONE_TIME_URL && (
          <a
            className="btn"
            href={ONE_TIME_URL}
            target="_blank"
            rel="noreferrer noopener"
          >
            Donate once
          </a>
        )}
        {MONTHLY_URL && (
          <a
            className="btn btn-primary"
            href={MONTHLY_URL}
            target="_blank"
            rel="noreferrer noopener"
          >
            Support monthly
          </a>
        )}
      </div>
      <span className="control-hint">
        Opens Stripe in a new tab. Payment is handled entirely by Stripe —
        Vigilume never sees your card details, and this NVR sends nothing about
        you to anyone. Monthly support can be cancelled any time from the receipt
        Stripe emails you.
      </span>
    </section>
  );
}
