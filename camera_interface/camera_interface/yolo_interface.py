import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLO  # Assuming YOLOv11 segmentation model support
import os
import numpy as np
class CameraYoloNode(Node):
    def __init__(self):
        super().__init__('camera_yolo_node')
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Navigate up to the workspace root directory
        workspace_dir = os.path.abspath(os.path.join(script_dir, '../../../../../..'))
        # Construct the full path to the YOLO model file in the src directory
        detection_path = os.path.join(workspace_dir, 'src', 'camera_interface', 'models', 'detection.pt')
        segmentation_path = os.path.join(workspace_dir, 'src', 'camera_interface', 'models', 'segmentation.pt')
        
        # Debugging statement to verify the path
        print(f"detection path: {detection_path}")
        print(f"segmentation path: {segmentation_path}")
        # Check if the model file exists
        if not os.path.exists(detection_path):
            raise FileNotFoundError(f"The model file does not exist in the expected path: {detection_path}")
        
        if not os.path.exists(segmentation_path):
            raise FileNotFoundError(f"The model file does not exist in the expected path: {segmentation_path}")
        self.detectionmodel = YOLO(detection_path)  # Load YOLOv11 model
        self.segmentationmodel = YOLO(segmentation_path)
        print("model path successfull")
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',# change the topic to /zed/zed_node/rgb/image_rect_color if using a zed camera
            self.image_callback,
            10
        )

        # OpenCV bridge
        self.bridge = CvBridge()
        self.get_logger().info("Camera YOLO Node Initialized with model: " + segmentation_path)
        
        ## comment the lines if using directly through the zed 2i camaera
        self.cap = cv2.VideoCapture(0)  # Access default laptop camera (use 0)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open the laptop camera.")
    
    
    def run(self):
        self.get_logger().info("Starting video feed. Press 'q' to exit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("Failed to capture frame from camera.")
                break

            try:
                # Run YOLO detection and segmentation
                results = self.detectionmodel(frame, conf =0.2)

                for result in results:
                    # Draw bounding boxes
                    for i,box in enumerate(result.boxes.xyxy):  # Bounding box coordinates
                        print(box)
                        x1, y1, x2, y2 = map(int, box[:4])
                        confidence = float(result.boxes.conf[i])
                        class_id = int(result.boxes.cls[i])

                        # Draw bounding box and label on the image
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{self.model.names[class_id]} {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                results = self.segmentationmodel(frame, conf = 0.2)

                # Overlay segmentation masks on the image
                if results[0].masks is not None:
                    for mask in results[0].masks.data:
                        # Convert mask to numpy format
                        mask = mask.cpu().numpy()
                        # Apply color to the mask
                        colored_mask = np.zeros_like(frame, dtype=np.uint8)
                        colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)  # Green channel
                        frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

                # Publish the segmented image
                # segmented_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                # self.publisher.publish(segmented_msg)
                # self.get_logger().info("Published segmented image.")


                    # Display the frame with detections
                cv2.imshow("YOLO Detection and Segmentation", frame)

                    # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                self.get_logger().error(f"Error processing frame: {str(e)}")

        self.cap.release()
        cv2.destroyAllWindows()
    def image_callback(self, msg):
        print("inside the function")
        try:
            # Convert ROS2 Image message to OpenCV image
            results = self.detectionmodel(msg, conf =0.2)

            for result in results:
                    # Draw bounding boxes
                for i,box in enumerate(result.boxes.xyxy):  # Bounding box coordinates
                    print(box)
                    x1, y1, x2, y2 = map(int, box[:4])
                    confidence = float(result.boxes.conf[i])
                    class_id = int(result.boxes.cls[i])

                    # Draw bounding box and label on the image
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{self.model.names[class_id]} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                results = self.segmentationmodel(frame, conf = 0.2)

                # Overlay segmentation masks on the image
                if results[0].masks is not None:
                    for mask in results[0].masks.data:
                        # Convert mask to numpy format
                        mask = mask.cpu().numpy()
                        # Apply color to the mask
                        colored_mask = np.zeros_like(frame, dtype=np.uint8)
                        colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)  # Green channel
                        frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

                # Publish the segmented image
                segmented_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                # self.publisher.publish(segmented_msg)
                self.get_logger().info("Published segmented image.")

            # Show the image
            cv2.imshow("YOLO Detection and Segmentation", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraYoloNode()
    # node.run() ##uncommonent to use local camera without publishing images to ros2
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
