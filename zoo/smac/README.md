## PYSC2 Env
LightZero use modified pysc2 env (for more maps and agent vs agent training), and you need to install ,
```powershell
# install starcraft
conda create -n ace python=3.8
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip SC2.4.10.zip
export SC2PATH="StarCraftII"
pip install pysc2 protobuf==3.19.5
```