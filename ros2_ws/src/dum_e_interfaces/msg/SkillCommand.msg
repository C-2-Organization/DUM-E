# dum_e_interfaces/msg/SkillCommand.msg

uint8 PICK=0
uint8 FIND=1
uint8 LOOK_AT=2          # ✅ 추가

# uint8 SHAKE=3
# uint8 HANDOVER_TO_USER=4
# ...

uint8 skill_type
string object_name
geometry_msgs/PoseStamped target_pose
string params_json

