# Explanation

After discussion, we decided to extract some abnormal numbers such as:

- **Amount**: Extracts dollar amounts, e.g., `$100.00`.
- **Tracking Numbers**: Extracts UPS and FedEx tracking numbers.
- **Postal Codes**: Extracts Canadian postal codes.
- **Phone Numbers**:
  - **Domestic**: Extracts Canadian and US phone numbers, e.g., `123-456-7890`, `+1 123 456 7890`.
  - **International**: Extracts international phone numbers excluding US and Canada, e.g., `+44 20 7123 4567`.

## Details

### Amount Extraction
- Regular expression: `\$\s?\d+\.?\d*`
- Matches dollar amounts with optional spaces.

### Tracking Numbers
- Regular expression: `\b1Z[A-Z0-9]{16}\b|\b(?!\+\d{11,14}\b)\d{12,15}\b`
- Matches UPS and FedEx tracking numbers.

### Postal Codes
- Regular expression: `\b[ABCEGHJKLMNPRSTVXY]\d[ABCEGHJKLMNPRSTVXY] ?\d[ABCEGHJKLMNPRSTVXY]\d\b`
- Matches Canadian postal codes.

### Phone Numbers
#### Domestic
- Regular expression: `\b(\+1[-\s]?)?\d{3}[-\s]\d{3}[-\s]\d{4}\b`
- Matches Canadian and US phone numbers.

#### International
- Regular expression: `\+(?!1)\d{1,3}\s?\d{1,14}([- \s]?\d{1,13})?`
- Matches international phone numbers excluding US and Canada.

**However, as our manually written code may not perfectly catch all abnormal_number features, this part may be 
updated to some exisiting package later**
