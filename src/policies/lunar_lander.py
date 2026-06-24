def lunar_lander_policy(obs):
    # obs: [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
    angle_target = obs[0] * 0.5 + obs[2] * 1.0  # target angle based on position and velocity
    angle_todo = (angle_target - obs[4]) * 0.5 - obs[5] * 1.0
    
    hover_todo = (obs[1] - 1.0) * 0.5 - obs[3] * 0.5  # control vertical velocity
    
    if obs[6] or obs[7]:  # legs touching ground
        angle_todo = 0
        hover_todo = -(obs[3]) * 0.5
    
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2  # main engine
    elif angle_todo < -0.05:
        return 3  # right engine
    elif angle_todo > 0.05:
        return 1  # left engine
    return 0  # do nothing