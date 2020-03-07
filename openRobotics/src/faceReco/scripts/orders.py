#!/usr/bin/env python
import roslib
roslib.load_manifest('faceReco')
import sys
import rospy
from std_msgs.msg import String
import pickle


def callback(data):
    if "FOLLOW" in data.data:
        count = 0
        while(count < len(names) and not (names[count].upper() in data.data)):
            count += 1
        if(count < len(names)):
            orders_follow.publish(names[count])

name_file = open("names","rb")
names = pickle.load(name_file)
name_file.close()

orders_follow = rospy.Publisher('/orders/follow',String,queue_size=10)
talk_sub = rospy.Subscriber('/grammar_data',String,callback)

rospy.init_node('orders_node', anonymous=True)

try:
    rospy.spin()
except rospy.ROSInterruptException:
    print("Shutting down")
