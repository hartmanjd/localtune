localtune friend setup
======================

Thanks for helping! This measures how fast AI models run on your PC, so localtune can
give everyone better speed guesses. Nothing personal is collected: only your hardware
(graphics card, CPU, RAM) and the speed measurements.

What you need
-------------
- A Windows PC with an NVIDIA graphics card.
- Up to about 120 GB of free disk space (less with a smaller graphics card).
- A few hours where the PC can stay on. Your graphics card will work hard, like
  when gaming.

What to do
----------
1. Unzip this folder: right-click the zip, choose "Extract All".
2. Double-click friend_setup.bat in the unzipped folder.
   If Windows says "Windows protected your PC", click "More info", then "Run anyway".
3. If it asks you to install Ollama (the program that downloads the models), install
   it from the page it opens, then double-click friend_setup.bat again.
4. Leave the PC alone until it says "All done". Please close games, videos and
   video calls while it runs.
5. Send Justin the file it puts on your Desktop:
   localtune-results-<your PC name>.jsonl

What it does, step by step
--------------------------
1. Checks for an NVIDIA graphics card.
2. Installs uv, a small tool that installs localtune (no admin rights needed).
3. Installs localtune from the file in this folder.
4. Checks that Ollama is installed and running.
5. Downloads the AI models that fit on your graphics card.
6. Installs llama.cpp, the engine that runs the models, and checks the PC is ready.
7. Runs each model and measures how fast it writes.

If the screen ever shows a GPU memory bar and seems stuck, press Enter to carry on.
You can close the window at any time. Double-click friend_setup.bat again and it
carries on where it stopped.

Removing it afterwards
----------------------
At the end it asks whether to remove the downloaded models and localtune. Type Y to
remove them. Your results file on the Desktop stays.
To remove them later instead, double-click friend_cleanup.bat.
It only removes what the setup added: models you had before, and Ollama itself, stay.
Ollama can be uninstalled like any other app (Settings > Apps).
