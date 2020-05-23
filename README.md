# Manipulability
This repository shows simple examples for manipulability learning, tracking and transfer tasks. 

These approaches offer the possibility of transferring posture-dependent task requirements such as preferred directions for motion and force exertion in operational space, which are encapsulated in manipulability ellipsoids. The proposed formulations exploit tensor-based representations and take into account that manipulability ellipsoids lie on the manifold of symmetric positive definite matrices.

## Examples description
### Learning
	- ManipulabilityLearning [1], [3]
		This code shows how a robot builds models to follow a desired Cartesian trajectory while matching a desired profile of manipulability ellipsoids over time. The learning framework is built on two GMMs, one for encoding the demonstrated Cartesian trajectories, and the other one for encoding the profiles of manipulability ellipsoids observed during the demonstrations. The former is a classic GMM, while the latter is a GMM that relies on an SPD-matrices manifold formulation. The desired trajectoried for reproducing the task are constructed with GMR based on the two GMM models. The demonstrations are generated with a 3-DoFs planar robot that follows a set of Cartesian trajectories. 
		To see how a robot can then reproduce the obtained desired trajectories, see the examples in the Transfer folder.

### Tracking
	- ManipulabilityTrackingMainTask [2], [3]
		This code shows how a robot can match a desired manipulability ellipsoid as the main task (no desired position) using the manipulability tracking formulation with the manipulability Jacobian (Mandel notation).

	- ManipulabilityTrackingSecondaryTask [2], [3]
		This code implements a manipulability tracking task as a secondary objective. Here, the robot is required to hold a desired Cartesian position as main task, while matching a desired manipulability ellipsoid as secondary task using the manipulability Jacobian formulation (Mandel notation). 

	- ManipulabilityTrackingControllerGains [2], [3]
		This code shows how a robot matches a desired manipulability ellipsoid as the main task (no desired position) using the formulation based the manipulability Jacobien (Mandel notation). The matrix gain used for the manipulability tracking controller is now defined as the inverse of a 2nd-order covariance matrix representing the  variability information obtained from a learning algorithm. 

	- ManipulabilityTrackingWithNullspace [3]
		This code shows how a robot can match a desired manipulability ellipsoid as the main task, while keeping a desired joint configuration as secondary task (by using the nullspace of the manipulability Jacobian).

	- ComDynamicManipulabilityTracking [3]
		This code shows how a robot can match a desired center-of-mass dynamic manipulability ellipsoid as a main task (no desired position) using the manipulability tracking formulation with the manipulability Jacobian (Mandel notation). The manipulability definition uses the Jacobian specified at the center of mass.

	- ManipulabilityActuationContribution [3]
		This code illustrates the effect of actuation contribution on the shape of the manipulability ellipsoid. A weight matrix representing the maximum joint velocity is added to the definition of the velocity manipulability.

	- ManipulabilityTrackingDualArmSystem [3]
		This code shows an example of manipulability tracking for a dual-arm system. The two robots work to match a desired dual-arm manipulability ellipsoid in a master-slave principle. The main task of the left robot is to match the desired manipulability ellipsoid, while the main task of the right robot is to keep its end-effector at the same position as the first robot and its secondary task is to match the desired manipulability.

### Transfer
	- ManipulabilityTransferWithManipulabilityJacobian [3]
		This code shows how a robot learns to follow a desired Cartesian trajectory while modifying its joint configuration to match a desired profile of manipulability ellipsoids over time. The learning framework is built on two GMMs, one for encoding the demonstrated Cartesian trajectories, and the other one for encoding the profiles of manipulability ellipsoids observed during the demonstrations. The former is a classic GMM, while the latter is a GMM that relies on an SPD-matrices manifold formulation. The tracking framework is built on the tracking formulation based on the manipulability Jacobian. The demonstrations are generated with a 3-DoFs planar robot that follows a set of Cartesian trajectories. The reproduction is carried out by a 5-DoF planar robot.

	- ManipulabilityTransferWithCostMinimization [1]
		This code shows how a robot learns to follow a desired Cartesian trajectory while modifying its joint configuration to match a desired profile of manipulability ellipsoids over time. The learning framework is built on two GMMs, one for encoding the demonstrated Cartesian trajectories, and the other one for encoding the profiles of manipulability ellipsoids observed during the demonstrations. The former is a classic GMM, while the latter is a GMM that relies on an SPD-matrices manifold formulation. The tracking framework is built on the the minimization of the Stein divergence between the current and desired manipulability ellipsoid. The demonstrations are generated with a 3-DoFs planar robot that follows a set of Cartesian trajectories. The reproduction is carried out by a 5-DoF planar robot.

## References
[1] Rozo, L., Jaquier, N., Calinon, S., and Caldwell, D. G. (2017). *Learning manipulability ellipsoids for task compatibility in robot manipulation.* In IEEE/RSJ Intl. Conf. on Intelligent Robots and Systems (IROS), pages 3183–3189. [pdf](https://leonelrozo.weebly.com/uploads/4/4/3/4/44342607/rozoiros17compressed.pdf)

[2] Jaquier, N., Rozo, L., Caldwell, D. G., and Calinon, S. (2018). *Geometry-aware tracking of manipulability ellipsoids.* In Robotics: Science and Systems (R:SS). [pdf](http://njaquier.ch/files/RSS18_Jaquier_final.pdf)

[3] Jaquier, N., Rozo, L., Caldwell, D. G., and Calinon, S. (2020). *Geometry-aware manipulability learning, tracking and transfer.* Available as arXiv:1811.11050. [pdf](http://njaquier.ch/files/ManipTransfer_arXiv.pdf)

### Authors
Noémie Jaquier and Leonel Rozo

http://njaquier.ch/

http://leonelrozo.weebly.com/

```
This source code is given for free! In exchange, we would be grateful if you cite the corresponding reference in any academic publication that uses this code or part of it:

[1] 
@INPROCEEDINGS{Rozo17IROS:ManTransfer,
	AUTHOR 		= {L. Rozo and N. Jaquier and S. Calinon and D. G. Caldwell},
	TITLE 		= {Learning Manipulability Ellipsoids for Task Compatibility in Robot Manipulation},
	BOOKTITLE	= IROS,
	YEAR 		= {2017},
	PAGES 		= {3183--3189}
}

[2] 
@INPROCEEDINGS{Jaquier18,
	AUTHOR		= {Jaquier, N and Rozo, L. and Caldwell, D. G. and Calinon, S.}, 
	TITLE		= {Geometry-aware Tracking of Manipulability Ellipsoids},
	BOOKTITLE	= R:SS,
	YEAR		= {2018},
	PAGES		= {}
}

[3] 
@ARTICLE{Jaquier20:IJRRManipulability,
	AUTHOR 		= {Jaquier, N and Rozo, L. and Caldwell, D. G. and Calinon, S.},
	TITLE   	= {Geometry-aware Manipulability Learning, Tracking and Transfer},
	JOURNAL 	= {arXiv preprint 1811.11050},
	YEAR    	= {2020},
	VOLUME  	= {},
	NUMBER  	= {},
	PAGES   	= {}
}
```

