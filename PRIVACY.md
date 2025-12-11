# Trust Layer – Browser Extension Privacy Policy

_Last updated: 12 Dec 2025_

Trust Layer is a browser extension that helps users quickly understand how a website handles their data by analyzing the website’s own privacy policy.

## What the extension does

When you click “Scan Privacy Policy”, Trust Layer:

1. Reads the content of the current page in your browser to find a link to the website’s privacy policy or terms.
2. Sends only the privacy policy URL to our backend service.
3. The backend downloads the policy text from that website and uses the OpenAI API to generate a summary and privacy score.
4. The result (score, risk label, and short bullets) is shown in the extension popup.

## Data we collect

The extension itself does **not** collect or store any of the following from users:

- Names, email addresses, or contact information  
- Passwords or authentication data  
- Payment or financial information  
- Personal communications (emails, chats, messages)  
- Files, form contents, or browsing history

The only information sent from the extension to our backend is:

- The URL of the privacy policy page to be analyzed.

## Server logs

Our backend service may temporarily log standard technical information such as:

- IP address  
- Timestamp  
- Requested URL (privacy policy URL)  

These logs are used only for debugging, reliability, and security, and are not sold or shared with third parties.

## Third-party services

We use the OpenAI API to analyze privacy policy text. The policy text and prompt are sent to OpenAI to generate a structured summary. No user identity or account details are included in this request.

OpenAI’s use of this data is governed by their own terms and privacy policy.

## Data sharing and sale

- We do **not** sell user data.  
- We do **not** share user data with advertisers or data brokers.  
- We do not use data for creditworthiness or lending decisions.

## Your choices

You can remove Trust Layer at any time from your browser’s extensions page. Once removed, the extension will no longer run or send any requests.

## Contact

If you have questions about this policy or Trust Layer, contact:  
v6190340@gmail.com
