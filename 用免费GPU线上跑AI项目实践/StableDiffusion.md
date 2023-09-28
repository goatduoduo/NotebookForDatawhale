# 部署stable-diffusion

最后一个任务，部署属于自己的AI绘画。但是由于这个模型过于巨大，使得笔记本极难部署，因此需要使用云服务器。


![新建项目](/img/virtaicloud14.png)

镜像大小=14GB，数据集大小=56.3GB

这个恐怖的79GB大小对于现在的硬盘来说可以承受，相当于一个大型游戏的大小了。

而运行项目变得格外简单，甚至不需要配置代码，直接配置端口即可。

![新建项目](/img/virtaicloud15.png)


接下来执行4步操作：

1. 解压代码及模型 
``` bash
tar xf /gemini/data-1/stable-diffusion-webui.tar -C /gemini/code/
``` 
2. 解压配置文件到隐藏目录/root/.cache 
``` bash
tar xf /gemini/data-1/cache.tar -C /root/ 
```
3. 拷贝frpc内网穿透文件 （注意有两行 -> 两条指令）
``` bash
cp /gemini/data-1/frpc_linux_amd64 /root/miniconda3/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2
chmod +x /root/miniconda3/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2
```
4. 运行项目 （注意有两行 -> 两条指令）
``` bash
cd /gemini/code/stable-diffusion-webui
python launch.py --deepdanbooru --share --theme dark --xformers --listen --gradio-auth qdy:123456
```

运行后的终端：

![新建项目](/img/virtaicloud16.png)

出现了 **Running on local URL:  http://0.0.0.0:7860** 之后，说明我们可以通过 **http://direct.virtaicloud.com:4xxxx/** 来进行访问，用户名和密码分别为qdy 123456

接下来就可以体验自己的stable diffusion了

![新建项目](/img/virtaicloud17.png)

说实话，我还用的不是特别熟练，因为自己也不知道要生成什么东西。

如果需要重复执行，只需要做上述第4步就可以了。

当然别做太大，否则会爆内存……

![新建项目](/img/virtaicloud18.png)

当然我完全不知道应该怎么生成提示词的时候，我借助了GPT的力量。

朵朵：当然现在我想生成不列颠的红衣军（火枪手），可以写关于这个的提示词么，并且用英文。

GPT：
当生成关于不列颠的红衣军（火枪手）的绘画时，以下是一些与这个主题相关的英文提示词：

1. Redcoats
2. British soldiers
3. Revolutionary War
4. Musketeers
5. Colonial era
6. Battle formation
7. Uniforms
8. Historical weaponry
9. Flagbearer
10. Regimental colors
11. Firing line
12. Fortifications
13. Battlefield
14. Officer's command
15. Bayonets
16. Cannon fire
17. Tactical maneuvers
18. 18th century warfare
19. Patriot resistance
20. Historical accuracy

这些提示词可以帮助你在绘画中捕捉不列颠红衣军（火枪手）在历史战场上的场景和氛围。希望这些提示词能够激发你的创作灵感！

![新建项目](/img/virtaicloud19.png)