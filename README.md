
# Jetson Obstacle Avoidance

A project to create a holistic obstacle avoidance 
system on a Jetson Nano, from image processing, 
creating a depth map (using D. Wofk and MIT's FastDepth) and 
segmented image (using SegNet), to providing feedback to a user through 
waist mounted motors in real time.



## Hardware Requirements

This project was conducted using an Nvidia Jetson Nano connected to a Raspberry Pi v2 
camera module, and has not been tested with other cameras. Pursue the functionality of alternative
image capture methods at your own peril. It might make it easier to work with the camera, rather than 
make a big GStreamer video object.
## Fresh Installation

Note: if using the Jetson in the lab with the existing installation, skip this section.

Before starting, it is recommended to set up a virtual environment to build all the packages
in, to prevent a host of issues. Setting one up is a simple process, detailed [here](https://docs.python.org/3/tutorial/venv.html).

If starting with a fresh Jetson install, 
first install the [Jetpack SDK](https://developer.nvidia.com/embedded/jetpack). 
This will set up most of the necessary libraries, 
as well as those specifically required for SegNet.
The model/network files must be moved into the dev directory under "networks".
They are not included in the repo as they're quite large.

For the dependencies of FastDepth, first install PyTorch. Then, follow the instructions
in their repo found [here](https://github.com/dwofk/fast-depth),
under the heading "Installing the TVM Runtime". 
Other required files (including those for TVM) are included in this repo.

A few other libraries are used that may not be installed by default.
They can be installed with:

```bash
python3 -m pip install --upgrade pip  (ensure pip is latest version)
python3 -m pip install opencv-python requests pillow
```

After that, clone this repo to a suitable dev workspace.


## Quick Start

If all has gone according to plan (or if you're using the Jetson in our lab),
you will be able to run the full script using the following commands:

```bash
venv (or whatever command is used to activate your particular virtual environment)
cd /your/dev/ws/ (done automatically on the lab's Jetson)
python3 testable.py
```
The program asks for the name to call video files (without extension), as well as
the number of frames an object must appear before sending an avoidance signal. Then,
output of setting up a GStreamer object, SegNet, and Fastdepth
will show in the terminal, and there is a significant wait on the first frame processing
before the model "warms up" to normal runtimes. After that, several image boxes should appear showing the camera feed.
Exit out with Q or Ctrl+C. After running, the videos will be saved in the same directory.

testable.py is the main file, and the one you'll likely be working with most. 
Some additional subroutines can be tested more simply using the files in the subdirectory
subscripts, like saving color video, or doing segmentation or depth mapping separately. Also 
notable is single_img.py, which will take only a single image, but do all the same processing
except for sending signals to the feedback system. 

Most of the code is commented, but obviously documentation tends to suffer in a one person project.
If you have questions, please reach out to me here or by email. 
## Next Steps

As the system currently runs, SegNet isn't accurate enough to produce good results.
We think this is due to the training of the network on the SUN-RGB-D dataset,
which has way more classes than we care about. 

So, our solution is to retrain the network using a simpler, smaller dataset
from where we would conduct testing. That's been done here in CVAT, and the login/password
for the account with those images is in the lab cabinet with the other Jetson stuff. 

To actually do the retraining, I was following [this guide](https://www.highvoltagecode.com/post/edge-ai-semantic-segmentation-on-nvidia-jetson/)
created by a Jetson community member. It's a bit old though, and I haven't been able to get it to work yet. We've been using
the UMaine ACG, but it's tough for them to get the particular dependencies to align correctly. I got the farthest
by doing it on a home computer and installing the exact versions specified in that guide.

If you encounter problems, I recommend asking on the Jetson community forums; they're super responsive.
Alternatively, you can email me at connor.firth@maine.edu.

Best of luck!
