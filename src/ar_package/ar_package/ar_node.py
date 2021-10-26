from numpy.lib.index_tricks import fill_diagonal
import cv_bridge
import rclpy
import sys
import cv2
from cv2 import aruco
from cv_bridge import CvBridge, CvBridgeError
from rclpy.logging import shutdown

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import matplotlib.pyplot as plt
import numpy as np
from rclpy.node import Node

class ar_node(Node):   

    node_name = "ar_node"
    ar_topic_name = "/ar_topic"
    id_topic_name = "/cmd_vel"
    recieve_topic_name = "/image_raw/compressed"
    ar_id = None
    #tmr_cnt = False

 #construct
    def __init__(self):
        super().__init__(self.node_name)  #node_name_configuration
        self.get_logger().info( "%s....now....initializing...."  % (self.node_name))  #plesse_make_instance_here
        self.bridge = CvBridge()
        self.vel = Twist()
        self.vel.linear.x = 0.0
        self.get_logger().info( "%s....finish....initilaizing" % (self.node_name) )
        
        self.img_subsc_ = self.create_subscription(CompressedImage, self.recieve_topic_name, self.call_back, 1)
        self.image_pub_ = self.create_publisher(Image, self.ar_topic_name, 1)
        self.id_pub_ = self.create_publisher(Twist, self.id_topic_name, 1)
        
        self.time_period = 0.5
        self.tmr = self.create_timer(self.time_period, self.id_pub_call_back)
        #self.cout = 0

#distruct
    def __del__(self):
        self.get_logger().info("%s....finish....run" % (self.node_name) )
        
#clean_window_func
    def cleanup(self):
        rclpy.shutdown
        cv2.destroyAllWindows()

#call_back_func
    def call_back(self, CompressedImage):
        try:
            img_buf = np.fromstring( bytes(CompressedImage.data), dtype=np.uint8)  #convert_to_ndarray
            img_deco = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)  #convert_opencv
            self.recognation_ar(img_deco)  #AR_recognaized_func
            self.image_pub_.publish(self.ar_img_msg)  #Publish

        except CvBridgeError as e:
           print(e)

    def id_pub_call_back(self):
       
        if self.ar_id == 0:
            #self.vel.linear.x = 0.25
            self.get_logger().info( "ID_number = %s" % (self.ar_id) )
            #self.get_logger().info( "published_linear.x = %s" % (self.vel.linear.x) )
        #else:
            #self.vel.angular.x = 0.0
            #self.get_logger().info( "published_linear.x = %s" % (self.vel.linear.x) )

        #self.id_pub_.publish(self.vel)


#ros2_img_to_opencv_and_recognation_AR_func 
    def recognation_ar(self, frame):
        ### --- aruco_configration --- ###
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)  #get_ar_dictionary_range
        parameters = aruco.DetectorParameters_create()  #create_ar_parameters

        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)  #resize_iamge_size
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        self.ar_img = aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))  #draw_marker
        self.ar_img = cv2.cvtColor(self.ar_img, cv2.COLOR_BGR2RGB)

        self.ar_img_msg = self.bridge.cv2_to_imgmsg(self.ar_img, "bgr8")  #ros2のImage型メッセージに変換
        self.ar_id = np.ravel(ids)

        #cv2.imshow('ar_img', self.ar_img)
        #cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)  #initilaizing_process
    node = ar_node()  #create_node_instance
    
    while rclpy.ok:
        rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown(ar_node.cleanup)
    
if __name__ == "__main__":
    main()

            #try:
            #node = ar_node()  #ノードの生成
            #rclpy.spin(node) #ノードの処理をループ実行

        #except KeyboardInterrupt:
            #node.destroy_node()  #ノードの破棄
            #rclpy.shutdown(ar_node.cleanup)
        #finally:
            # プロセスの終了
            #node.destroy_node()
            #rclpy.shutdown(ar_node.cleanup)