# StyLit: Illumination-Guided Stylization of 3D Renderings

This repository contains the original source code of StyLit [[7]](#references) taken from [here](https://github.com/jamriska/ebsynth). Citation can be found [below](#citation).


## Basic usage

### Build

Run one of the build scripts, depending on your current system. Executable should be created in the bin folder.

### Run

```
PATH=examples/1;
./bin/ebsynth -style $PATH/source_style.png
-guide $PATH/source_fullgi.png $PATH/target_fullgi.png -weight 0.66
-guide $PATH/source_dirdif.png $PATH/target_dirdif.png -weight 0.66
-guide $PATH/source_indirb.png $PATH/target_indirb.png -weight 0.66
-output output/output.png
```

  #### Options
  ```
  -style <style.png>
  -guide <source.png> <target.png>
  -weight <value>
  -uniformity <value>
  -patchsize <value>
  -pyramidlevels <number>
  -searchvoteiters <number>
  -patchmatchiters <number>
  -extrapass3x3
  -backend [cpu|cuda]
  ```

## Benchmarks

<table>
  <thead>
    <tr>
      <th>System</th>
      <th>Stylit Params</th>
      <th>Benchmark</th>
      <th>Output</th>
  </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=3>Intel® Core™ i7-3537U CPU @ 2.00GHz × 4 </br> Ubuntu 14.04 LTS </td>
      <td rowspan=3>uniformity: 3500 </br> 
                    patchsize: 5 </br> 
                    pyramidlevels: 7 </br> 
                    searchvoteiters: 6 </br> 
                    patchmatchiters: 4 </br> 
                    stopthreshold: 5 </br> 
                    extrapass3x3: no </br> 
                    backend: cpu </br>
                    weight 0.66 </br>
                    build: linux-cpu_only </br>
                    log id: #100324</td>
      <td>time(1)</td>
      <td> 1 min 12.74 sec </br> verbose: time(1)_examples_1-id_100324.log </td>
    </tr>
    <tr>
      <td rowspan=1>pmap(1)</td>
      <td> total kB   (Kbytes)174152  (RSS)126104  (Dirty)124612 </br> verbose : pmap_examples_1-id_100324.log</td>
    </tr>
    <tr>
      <td rowspan=1>gprof</td>
      <td> verbose : gprof_examples_1-id_100324.log</td>
    </tr>
    <tr>
      <td rowspan=3>Intel® Core™ i7-8700K CPU @ 3.70GHz × 12 </br> Ubuntu 18.04.3 LTS </br> GeForce GTX 1080 Ti/PCIe/SSE2 </td>
      <td rowspan=3>uniformity: 3500 </br> 
                    patchsize: 5 </br> 
                    pyramidlevels: 7 </br> 
                    searchvoteiters: 6 </br> 
                    patchmatchiters: 4 </br> 
                    stopthreshold: 5 </br> 
                    extrapass3x3: no </br> 
                    backend: cpu </br>
                    weight 0.66 </br>
                    build: linux-cpu+cuda</td>
      <td>time</td>
      <td> 0 min 1.752 sec </br> verbose: time_gpu_examples_1.log </td>
    </tr>
    <tr>
      <td rowspan=1>pmap</td>
      <td> verbose : pmap_gpu_examples_1.log </td>
    </tr>
    <tr>
      <td rowspan=1>gprof</td>
      <td> cannot compile with profiler options using nvcc</td>
    </tr>
  </tbody>
</table>

#### Usage:

- [`time(1)`](http://man7.org/linux/man-pages/man1/time.1.html) :
  - Run `$ time ./bin/ebsynth ....`  (refer to [#run](#run) for full list of arguments).
- [`pmap(1)`](http://man7.org/linux/man-pages/man1/pmap.1.html) :
  - Run `$ ./bin/ebsynth ...`    (refer to [#run](#run) for full list of arguments).
  - While the above is running, open a new tab and run `$ pid_ebsynth="$(pgrep ebsynth)"; pmap -X $pid_ebsynth`
- [`gprof`](http://man7.org/linux/man-pages/man1/gprof.1.html) :
  - Add `-Wall -pg -O1` flag in the build script like in `build-linux-cpu_only_gprof.sh`.
  - Compile.
  - Run `$ ./bin/ebsynth ...`    (refer to [#run](#run) for full list of arguments). A `gmon.out` file should be generated.
  - Run `$ gprof ./bin/ebsynth` (you don't have to provide arguments here).

--------------------------------------------------------------------------

## License

The code is released into the public domain. You can do anything you want with it.

However, you should be aware that the code implements the PatchMatch algorithm, which is patented by Adobe (U.S. Patent 8,861,869). Other techniques might be patented as well. It is your responsibility to make sure you're not infringing any patent holders' rights by using this code. 

## Citation

If you find this code useful for your research, please cite:

```
@misc{Jamriska2018,
  author = {Jamriska, Ondrej},
  title = {Ebsynth: Fast Example-based Image Synthesis and Style Transfer},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jamriska/ebsynth}},
}
```

## References

1. Image Analogies  
Aaron Hertzmann, Chuck Jacobs, Nuria Oliver, Brian Curless, David H. Salesin  
In SIGGRAPH 2001 Conference Proceedings, 327–340.  
2. Texture optimization for example-based synthesis  
Vivek Kwatra, Irfan A. Essa, Aaron F. Bobick, Nipun Kwatra  
ACM Transactions on Graphics 24, 3 (2005), 795–802.  
3. Space-Time Completion of Video  
Yonatan Wexler, Eli Shechtman, Michal Irani  
IEEE Transactions on Pattern Analysis and Machine Intelligence 29, 3 (2007), 463–476.  
4. PatchMatch: A randomized correspondence algorithm for structural image editing  
Connelly Barnes, Eli Shechtman, Adam Finkelstein, Dan B. Goldman  
ACM Transactions on Graphics 28, 3 (2009), 24.  
5. Self Tuning Texture Optimization  
Alexandre Kaspar, Boris Neubert, Dani Lischinski, Mark Pauly, Johannes Kopf  
Computer Graphics Forum 34, 2 (2015), 349–360.  
6. LazyFluids: Appearance Transfer for Fluid Animations  
Ondřej Jamriška, Jakub Fišer, Paul Asente, Jingwan Lu, Eli Shechtman, Daniel Sýkora  
ACM Transactions on Graphics 34, 4 (2015), 92.  
7. StyLit: Illumination-Guided Example-Based Stylization of 3D Renderings  
Jakub Fišer, Ondřej Jamriška, Michal Lukáč, Eli Shechtman, Paul Asente, Jingwan Lu, Daniel Sýkora  
ACM Transactions on Graphics 35, 4 (2016), 92.  
8. Example-Based Synthesis of Stylized Facial Animations  
Jakub Fišer, Ondřej Jamriška, David Simons, Eli Shechtman, Jingwan Lu, Paul Asente, Michal Lukáč, Daniel Sýkora  
ACM Transactions on Graphics 36, 4 (2017), 155.  
