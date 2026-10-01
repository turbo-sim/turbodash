
- Fix bug where turbodash is relaunching everytime
- ~~Hide the the aspect ratio in the web app when using radial turbines~~ - Done
- We should move the turbogrid export funtionality into its own module of the source code
  - We should have some options in the yaml configuration file to configure the turbogrid export
  - For example, the number of blade sections (now its 10), or the number of points per section (resolution)
  - Now it seems that some bladegen config files are being generated, we do not need to create any bladegen files
  - ~~We should have in the app an option to export the turbogrid generation files for the blades that we want, for example to export the files for the stator, the rotor or both. Also in multistage cases.~~ - Done
  - We should find a nice way to specify the axial location of subsequent blades in axial turbines, now I think that its only a visual thing in the plot blades functions
  - How does turbogrid process multirow cases? Is it a single hub/shourd file for the entire machine and several blade files? or is it one hub and shroud file per blade?
  - ~~We should have a buttom to export the turbogrid files from the web app (as a zip folder?)~~ - Done
  - ~~Add hub to tip ratio at the exit of the row to the geometry table in the app~~ - Done
 