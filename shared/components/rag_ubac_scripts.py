from shared.configs.static import VALID_ROLES

def get_ubac_role():
    """Prompt user to select their role for UBAC access control."""
    print("\n=== Role-Based Access Control ===")
    print("Please select your role to determine document access:")
    print("1. executive - Full access to all documents")
    print("2. hr - Access to HR policies and onboarding (no executive strategy)")
    print("3. junior - Access only to onboarding guide")
    
    role, _try = "", 0
    while role not in VALID_ROLES and _try < 3:
        role = input("Enter your role (executive/hr/junior): ").strip().lower()
        if role not in VALID_ROLES:
            print("Invalid role. Please choose one of: executive, hr, junior.")
            if _try == 2:
                print("You are exceeded maximum number of attempt; please refer the persona documentation for more referenece")
                return None
            _try += 1
    return role

def display_access_info(role):
    """Display what documents the user can access."""
    if role == "executive":
        print("✓ Executive Strategy")
        print("✓ HR Policies and Benefits") 
        print("✓ Onboarding Guide")
    elif role == "hr":
        print("✗ Executive Strategy (restricted)")
        print("✓ HR Policies and Benefits")
        print("✓ Onboarding Guide")
    elif role == "junior":
        print("✗ Executive Strategy (restricted)")
        print("✗ HR Policies and Benefits (restricted)")
        print("✓ Onboarding Guide")
    else:
        print(f"No details found for the role: {role}")