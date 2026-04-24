"""
One-time script to seed default raga profiles into Firebase Firestore.

Usage:
  python seed_firebase.py

Requires FIREBASE_SERVICE_ACCOUNT_PATH or FIREBASE_CREDENTIALS_JSON
in your .env file (or environment).
"""

from dotenv import load_dotenv
load_dotenv()

from db.firebase import seed_default_ragas, get_all_ragas

if __name__ == "__main__":
    print("Seeding default ragas...")
    created = seed_default_ragas()
    if created:
        print(f"Created {len(created)} raga(s): {', '.join(created)}")
    else:
        print("All default ragas already exist — nothing to seed.")

    ragas = get_all_ragas()
    print(f"\nFirestore now contains {len(ragas)} raga(s):")
    for r in ragas:
        print(f"  [{r['id']}] {r['name']} — {r.get('difficulty', '?')} — params: {len(r.get('pitch_params', {}))}")
