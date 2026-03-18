//
// Created by kunwoo on 24. 7. 23.
//

#ifndef PERCEPTION_LIDAR_TYPES_H
#define PERCEPTION_LIDAR_TYPES_H

#include <iostream>
#include <vector>

struct Box {
    float x, y, w, h, angle, confidence, class_type, id, velocity_x=0, velocity_y=0, velocity_z=0;
};

#endif //PERCEPTION_LIDAR_TYPES_H
