项目要求：
1.视觉输入，占据栅格地图表示，地图四种元素障碍物，可行走区域，自车，以及目标停车位，目前是三通道的二值图像。
2.低维运动信息输入：自车运动信息（x,y,vx,vy,cos(yaw),sin(yaw)）
                目标车位信息（X,Y,0，0，1，0）,就是一个goal state
                
要求在避开障碍物的同时停在指定车位上
每回合障碍物的位置随机，并且目标停车位的位置随机，但是一定是在标注好的车位上。
最终期望：
能够停在接近车位的附近，误差允许范围已写在环境的_is_success判断中，可以略微调整，但是图像上要差不多
