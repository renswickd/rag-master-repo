from shared.configs.static import VALID_ROLES 

def get_ubac_role():
    # Prompt for role/persona
    role, _try = "", 0
    while role not in VALID_ROLES and _try < 3:
        role = input("Enter your role (executive/hr/junior): ").strip().lower()
        if role not in VALID_ROLES:
            print("Invalid role. Please choose one of: executive, hr, junior.")
            _try += 1
        if _try == 2:
            print("You are exceeded maximum number of attempt; please refer the persona documentation for more referenece")
            return None
    
    return role