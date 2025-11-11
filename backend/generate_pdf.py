# ...existing code...

try:
    with open("prediction_results.txt", "r") as f:
        prediction_result = f.readline().strip()
        confidence_level = f.readline().strip()
        raw_output = f.readline().strip()
        enhanced_prob = f.readline().strip()
        model_status = f.readline().strip()
except FileNotFoundError:
    prediction_result = "Error"
    confidence_level = "0.00%"
    raw_output = "0.0000"
    enhanced_prob = "0.0000"
    model_status = "Error"

# Update PDF cells with actual values
pdf.cell(0, 10, f"PREDICTION: {prediction_result}", ln=True, align='C')
pdf.cell(0, 10, txt=f"Prediction Result: {prediction_result}", ln=True)
pdf.cell(0, 10, txt=f"Confidence Level: {confidence_level}", ln=True)
pdf.cell(0, 10, txt=f"Raw Model Output: {raw_output}", ln=True)
pdf.cell(0, 10, txt=f"Enhanced Probability: {enhanced_prob}", ln=True)
pdf.cell(0, 10, txt=f"Model Status: {model_status}", ln=True)