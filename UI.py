import joblib
import pandas as pd

mlModel = "diabetes_logreg_pipeline.joblib"  # change if your filename is different

# Display a numbered menu and return the selected option
def pick_one(prompt: str, options: list[str]) -> str:
    print(f"\n{prompt}")
    # Print numbered options
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    # Keep asking until valid input is given
    while True:
        choice = input("Choose number: ").strip()
        # Check that input is a valid number within range
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice. Try again.")

# Ask user for an integer within a given range
def pick_int(prompt: str, min_v: int, max_v: int) -> int:
    while True:
        # Get user input
        s = input(f"{prompt} [{min_v}-{max_v}]: ").strip()
        # Check if input is numeric
        if s.isdigit():
            v = int(s)
            # Validate range
            if min_v <= v <= max_v:
                return v
        print("Invalid number. Try again.")

# Ask user for a float within a given range
def pick_float(prompt: str, min_v: float, max_v: float) -> float:
    while True:
        # Get user input
        s = input(f"{prompt} [{min_v}-{max_v}]: ").strip()
        try:
            # Try converting to float
            v = float(s)
            # Validate range
            if min_v <= v <= max_v:
                return v
        except ValueError:
            pass # Ignore invalid conversion
        print("Invalid number. Try again.")


def main():
    # Load saved pipeline (preprocess + model)
    pipe = joblib.load(mlModel)
    print(f"Loaded model: {mlModel}")

    # Mode selection when the UI.py is runned in terminal
    mode = pick_one("Choose input mode", ["Manual input", "Demo diabetes female", "Demo no diabetes male"])

    # Collect inputs automatically
    if mode == "Demo diabetes female":
        age = 64
        hypertension = 1
        heart_disease = 0
        bmi = 28.24
        hba1c = 8.8
        glucose = 155
        gender = "Female"
        smoking = "current"
    
    # Collect inputs automatically
    elif mode == "Demo no diabetes male":
        age = 16
        hypertension = 0
        heart_disease = 0
        bmi = 27.32
        hba1c = 6.1
        glucose = 160
        gender = "Male"
        smoking = "No Info"

    # Collect inputs manually
    else:
        age = pick_int("Age", 0, 120)
        hypertension_choice = pick_one("Hypertension?", ["0 (No)", "1 (Yes)"])
        hypertension = int(hypertension_choice[0])
        heart_choice = pick_one("Heart disease?", ["0 (No)", "1 (Yes)"])
        heart_disease = int(heart_choice[0])
        bmi = pick_float("BMI", 0.0, 100.0)
        hba1c = pick_float("HbA1c level", 0.0, 20.0)
        glucose = pick_int("Blood glucose level", 0, 500)
        gender = pick_one("Gender", ["Female", "Male", "Other"])
        smoking = pick_one(
            "Smoking history",
            ["No Info", "never", "current", "former", "ever", "not current"]
        )

    # Convert to the exact feature names your pipeline expects (raw columns!)
    x = pd.DataFrame([{
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "gender": gender,
        "smoking_history": smoking,
    }])

    # Predict use the trained model here to predict the result
    probability = pipe.predict_proba(x)[0, 1]
    prediction = pipe.predict(x)[0]

    # Print results
    print("\n--- Result ---")
    print("Input:")
    print(x.to_string(index=False))
    print(f"\nPredicted class: {'Diabetes' if prediction == 1 else 'No diabetes'}")
    print(f"Probability of diabetes: {probability:.2%}")


if __name__ == "__main__":
    main()