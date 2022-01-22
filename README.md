# Vision-for-Agriculture-Competition
A project for Agriculture vision task based on double U-net.<br>
Please see and directly use the `Main.ipynb` to train our model, we use Google Colabatory to train. <br>
Our model architecture shows below:
<div align=center>
<img src=https://github.com/Ys-Jia/Vision-for-Agriculture-Competition/blob/main/Architecture.png height='400' width='800'>
</div> <br>
We separate the labels into three groups. The first one is only ground because it has the most number of labels, and the second group is label 1-4(cloud shadow, double plant, planter skip, standing water) which are the fewest four labels, and the final group is label 5-6(water way, weed cluster), which have same number of labels. By using three groups we could set three U-nets to mainly focus on the subset of labels respectively. The reason that we use this structure could be divided as two parts—First, each net could focus on one group that have relatively balanced labels, and the backward gradient will not be messed up, which means each group will have their own gradient to update parameters. Second, for different labels and U-net we could choose different loss function to enlarge the precision to further balance labels. Therefore, we have large flexibility for loss function choice and improving the performance of model.
To speed up the model convergence and control our model not to overfit, we also add the batch normalize layer and dropout layers in our U-net[4].<br>

Our model MIOU score is `35.2%` by the default setting in `Main.ipynp`! <br>
<br>
The sample prediction picture in validation set shows below (left is `prediction`, right is `Ground Truth`): <br>

<div align=center>
<img src=https://github.com/Ys-Jia/Vision-for-Agriculture-Competition/blob/main/Prediction_train.png height='400' width='800'>
</div> <br>

The test curve:
<div align=center>
<img src=https://github.com/Ys-Jia/Vision-for-Agriculture-Competition/blob/main/Test_curve.png height='400' width='800'>
</div> <br>
