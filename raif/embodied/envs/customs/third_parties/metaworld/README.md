# Metaworld

Customized version
* Because metaworld does not have support for custom images, I have to rewrite the Mujoco rendering classes. `raif.envs.customs.mujoco_env_custom_cam.py` => modify in Metaworld: `metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env.py` (every place marked with `# NOTE: ...`)

* I also have to change all the initialization method for all env class.

