# IS-MVSNet (ECCV 2022)

Our paper has been accepted as a conference paper in ECCV 2022!

ISMVSNet, a.k.a. Importance-sampling-based MVSNet, is a simple yet effective multi-view reconstruction method. 

This repo is maintained by the first author of IS-MVSNet. 

We will first release the Mindspore-based implementation through the company's official website. Our results demonstrated in the paper are obtained based on the Pytorch implementation. Due to the difference in operator behaviors and the noise introduced during weight transferring, these two versions may slightly differ in performance. Limited by the company's requirements, the Pytorch version is not yet releasable. Thus, you may first **star** and **watch** this repo before checking the Mindspore version so that you can access the coming Pytorch implementation in the first place.

## The main structure of IS-MVSNet
![alt](imgs/IS-MVSNet%20Framework%20Redraw.png "The main structure of IS-MVSNet")

## Error distribution estimation module
![alt](imgs/Photoconsistency%20loss.png "Error distribution estimation module")

## Depth selection module
![alt](imgs/Sampling%20interval%20illustrate.png "Depth selection module")

<!-- <div  align="center">  
<img src="https://github.com/NoOneUST/IS-MVSNet/blob/main/imgs/IS-MVSNet%20Framework%20Redraw.png" width="100%" alt="The main structure of IS-MVSNet" align="center" />
  
  The main structure of IS-MVSNet
</div>

<br/>

<div  align="center">
<img src="https://github.com/NoOneUST/IS-MVSNet/blob/main/imgs/Photoconsistency%20loss.png" width="40%" alt="Error distribution estimation module" align="center" />
  
  Error distribution estimation module
</div>

<br/>

<div  align="center">
<img src="https://github.com/NoOneUST/IS-MVSNet/blob/main/imgs/Sampling%20interval%20illustrate.png" width="60%" alt="Depth selection module" align="center" />
  
  Depth selection module
</div> -->
