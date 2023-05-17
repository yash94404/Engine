# Engine
**Per-Frame Face Tracking**

MTCNN face detection module from facenet-pytorch tracks the faces in each frame and finds the bounding boxes and facial landmarks for each face. The module works with nearly real-time efficiency and accuracy.

**Unique Face Detection and Person Tokenization per Video**

I used the Inception-ResnetV1 model trained on the VGGFace2 Dataset which finds the vector embedding for each face. We detect unique faces by finding the euclidean distance between vector embeddings across frames. If they are above the standard MTCNN threshold (0.7), then they are classified as different faces.

**Output List of Unique Faces in a Given Video and the Timestamps of Where Each Person Appears**

I then compiled the times in which each unique face’s vector embedding first appeared in the video into a list of timestamps and a list of corresponding faces from each vector embedding formed by the bounding box. For tokenization, we’ll assign a unique ID to each face.

**Performance Considerations**

I utilized the facenet-pytorch model as it gives us state-of-the-art face recognition performance while just using 128 bytes per face.  It directly understands the relationship between facial images and a condensed Euclidean space, where the distances accurately represent the level of similarity between faces. This allows us to use the Euclidean distance method to compare two faces. The facenet-pytorch model relies on MTCNN which uses cascaded convolutional neural networks and uses the correlation between face detection and alignment to improve efficiency. 

**Future Improvements/Ideas**
- Can improve the speed by sampling less frames (eg. sampling every 3rd frame such as the approach used by Fast MTCNN) and then performing face detection and tokenization on these sampled frames and extrapolating them to all frames
- Export PyTorch model to ONNX (https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- Possibly use a K-Nearest-Neighbors (KNN) clustering approach where for each image embedding, we classify it as an already stored face if it is close enough to another cluster of similar embeddings
- Have a dynamic threshold for the Euclidean distance so that tokenization would be more accurate

