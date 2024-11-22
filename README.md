# Lane detection based on car detection & tracking

## results
### road proposal

- notice: this proposal isn't precise and also in process of being completed  
![Demo](results/output.gif)

### lane detect with canny edge detection
  *in progress*  
![Demo](results/canny_edge_road_detect.png)

## methods
  1. make hitmap based on driving trajectory(considered direction)
   ![Demo](results/collected_points.png)
  2. applying gaussian filter to make wider distribution of pixel
   ![Demo](results/gaussian.png)
  3. identifying main roads using DBSCAN clustering
   ![Demo](results/main_road2.png)
  4. applying edge detection algorithm(in this case canny edge)
   ![Demo](results/canny_edge_road_detect.png)


