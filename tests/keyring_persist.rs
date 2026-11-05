//! Confirms keys land in Windows Credential Manager, not the in-memory mock.

#[test]
fn windows_credential_persists_across_entries() {
    let service = "VisionLab";
    let user = "VisionLab/visionlab-keyring-selftest";
    let first = keyring::Entry::new(service, user).expect("open credential");
    let _ = first.delete_credential();
    first
        .set_password("sk-test-visionlab")
        .expect("store password");
    drop(first);

    let second = keyring::Entry::new(service, user).expect("reopen credential");
    let stored = second.get_password().expect("read password");
    let _ = second.delete_credential();
    assert_eq!(stored, "sk-test-visionlab");
}
