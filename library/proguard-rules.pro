# Onnx runtime rules
-keep class ai.onnxruntime.** { *; }

# TensorFlow Lite; fixes "Couldn't find TensorFlow Lite runtime's InterpreterFactoryImpl class"
-keep class com.google.android.gms.tflite.** { *; }