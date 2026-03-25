def parse_label(label):
    parts = label.split("_")
    crop = parts[0]
    if len(parts) > 1:
        disease = "_".join(parts[1:])
    else:
        disease = "Unknown"
    return crop, disease
def get_remedy(label):
    crop, disease = parse_label(label)
    disease_lower = disease.lower()
    if "healthy" in disease_lower:
        return f"{crop} is healthy. Maintain watering and sunlight."
    if "blight" in disease_lower:
        return f"{crop}: Remove infected leaves. Use fungicides like Mancozeb."
    if "bacterial" in disease_lower:
        return f"{crop}: Use copper sprays. Avoid water splashing."
    if "virus" in disease_lower:
        return f"{crop}: Remove infected plants. Control insects like aphids."
    if "mold" in disease_lower:
        return f"{crop}: Reduce humidity. Improve airflow."
    if "spot" in disease_lower:
        return f"{crop}: Apply fungicide and prune affected areas."
    if "mite" in disease_lower:
        return f"{crop}: Use neem oil or miticides."
    return f"{crop}: Maintain proper care and monitoring."