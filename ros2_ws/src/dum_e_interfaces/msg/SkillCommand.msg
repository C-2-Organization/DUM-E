# dum_e_interfaces/msg/SkillCommand.msg

uint8 PICK=0
# uint8 HOLD=1
# uint8 SHAKE=2
# uint8 HANDOVER_TO_USER=3
# ...

uint8 skill_type
string object_name
geometry_msgs/PoseStamped target_pose
string params_json # {"hand":"left","intensity":0.8} 같은 옵션
