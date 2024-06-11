import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import serial
import time

def main():
    # Load the TensorFlow Lite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path='models/fmnist_full_quant.tflite')
    interpreter.allocate_tensors()

    # Get input details and quantization parameters
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    # Load the CIFAR-10 images and labels
    x_test = np.load('x_test_cifar.npy')
    y_test = np.load('y_test_cifar.npy').squeeze()

    # Define class names for CIFAR-10
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Initialize serial communication
    ser = serial.Serial(port='COM9', baudrate=115200, timeout=10)
    ser.flushInput()
    ser.flushOutput()

    # Select a single image and its label
    image_index = 4  # Use the first image for testing
    test_image = x_test[image_index]
    class_idx = y_test[image_index]
    random_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    # Quantize the image for the TFLite model
    quantized_image = np.array(test_image / input_scale + input_zero_point, dtype=np.uint8)

    # Local inference
    interpreter.set_tensor(input_details[0]['index'], [quantized_image])
    interpreter.invoke()
    local_output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Local predictions:{local_output}")
    local_predicted_class = np.argmax(local_output)

    # Send the image to the MCU
    # ser.write(test_image.tobytes())
    ser.write(quantized_image.flatten())
    time.sleep(1)  # Allow time for MCU to process the image

    # Read the image back from the MCU
    received_image_data = ser.read(32 * 32 * 3)  # Assuming 3 bytes per pixel for RGB
    received_image = np.frombuffer(received_image_data, dtype=np.uint8).reshape(32, 32, 3)

    # Display both images for comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(received_image)
    plt.title('Received Image from MCU')
    plt.show()

    time.sleep(1)
    # Read the prediction from the MCU
    pred = ser.read(10)  # Assume MCU sends back 10 bytes for class probabilities
    pred = np.frombuffer(pred, dtype=np.uint8)
    print(f"MCU predictions:{pred}")
    mcu_predicted_class = np.argmax(pred)
    print(f'MCU Predicted Class: {classes[mcu_predicted_class]}')

    # Compare local and MCU predictions
    if local_predicted_class == mcu_predicted_class:
        print("Success! Both predictions match.")
    else:
        print("Mismatch in predictions. Local: {}, MCU: {}".format(classes[local_predicted_class], classes[mcu_predicted_class]))

    # Close the serial port
    ser.close()

if __name__ == '__main__':
    main()
