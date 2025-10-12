from utils.inference import infer_image, infer_batch

# Single image prediction example
#single_result = infer_image("/Users/shreyamitra/ultrasound_classifier/data/images/Kidney-with-angiomyolipoma-uncoloured-3.jpg")
single_result = infer_image("data/images/acute pancreatitis with kidney left.png")
print("Result:", "abnormal" if single_result == 1 else "normal")

# Batch prediction example
batch_results = infer_batch("data/images/")
print(batch_results)
