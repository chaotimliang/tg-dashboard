"""
First-time setup for Telegram API access.
Run this once to authenticate with your Telegram account.

To get API credentials:
1. Go to https://my.telegram.org
2. Log in with your phone number
3. Go to "API development tools"
4. Create a new application (any name/description)
5. Copy the api_id and api_hash
"""
from telegram_client import save_credentials, get_client


def main():
    print("=" * 60)
    print("Telegram API Setup")
    print("=" * 60)
    print()
    print("To get API credentials:")
    print("1. Go to https://my.telegram.org")
    print("2. Log in with your phone number")
    print("3. Go to 'API development tools'")
    print("4. Create a new application")
    print("5. Copy the api_id and api_hash")
    print()

    # Get API credentials
    try:
        api_id = int(input("Enter your api_id: ").strip())
    except ValueError:
        print("Error: api_id must be a number")
        return

    api_hash = input("Enter your api_hash: ").strip()
    if not api_hash:
        print("Error: api_hash is required")
        return

    # Save credentials
    save_credentials(api_id, api_hash)
    print("\nCredentials saved!")

    # Connect and authenticate
    print("\nConnecting to Telegram...")
    client = get_client()

    if client.connect():
        print("Already authenticated! Setup complete.")
        client.disconnect()
        return

    # Need to authenticate
    print("\nPhone authentication required.")
    phone = input("Enter your phone number (with country code, e.g., +1234567890): ").strip()

    if not phone:
        print("Error: Phone number is required")
        return

    print("\nSending verification code...")
    try:
        phone_code_hash = client.send_code(phone)
    except Exception as e:
        print(f"Error sending code: {e}")
        client.disconnect()
        return

    print("Code sent!")
    code = input("Enter the code you received: ").strip()

    if not code:
        print("Error: Code is required")
        client.disconnect()
        return

    print("\nVerifying...")
    if client.sign_in(phone, code, phone_code_hash):
        print("\n" + "=" * 60)
        print("SUCCESS! You are now authenticated.")
        print("You can now run the Telegram viewer.")
        print("=" * 60)
    else:
        print("\nAuthentication failed. Please try again.")

    client.disconnect()


if __name__ == "__main__":
    main()
