

# 1) SQL Injection via string formatting
def insecure_sql_lookup(user_input, db_cursor):
    # Vulnerable: user input concatenated directly into SQL query
    query = f"SELECT * FROM users WHERE username = '{user_input}'"  # nosec
    db_cursor.execute(query)
    return db_cursor.fetchall()


# 2) Command Injection via shell=True and unsanitized input
def insecure_shell_run(user_cmd):
    import subprocess
    # Vulnerable: executes arbitrary commands from user input
    return subprocess.check_output(user_cmd, shell=True)  # nosec


# 3) Use of eval on untrusted input
def insecure_eval(expression_str):
    # Vulnerable: arbitrary code execution
    return eval(expression_str)  # nosec


# 4) Hardcoded credentials and weak crypto
HARDCODED_API_KEY = "sk_live_1234567890abcdef"  # nosec

def weak_password_hash(password):
    import hashlib
    # Vulnerable: fast, broken hash function for passwords
    return hashlib.md5(password.encode()).hexdigest()  # nosec


# 5) Insecure deserialization of untrusted data
def insecure_deserialize(pickled_bytes):
    import pickle
    # Vulnerable: arbitrary code execution during deserialization
    return pickle.loads(pickled_bytes)  # nosec


# 6) Disabling TLS verification in HTTP request
def insecure_http_get(url):
    import requests
    # Vulnerable: SSL verification disabled
    resp = requests.get(url, verify=False)  # nosec
    return resp.text


# 7) Path traversal when writing files
def insecure_file_write(filename, content):
    # Vulnerable: unvalidated path from user input
    with open(filename, "w") as f:  # nosec
        f.write(content)
    return True


# 8) Information leakage in exception messages
def insecure_error_leak(secret_value):
    try:
        raise ValueError("Simulated failure")
    except Exception as exc:  # broad except
        # Vulnerable: leaking sensitive value and full exception details
        return f"Error: {exc}; secret={secret_value}"  # nosec


