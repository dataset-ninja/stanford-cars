firs par


# Motivation

The authors' motivation in creating this dataset stemmed from the recognized potential of three-dimensional representations in computer vision, which have historically been regarded as the ultimate goal due to their promise of providing more accurate and concise depictions of the visual world compared to traditional view-based representations. While recent advancements have showcased the advantages of 3D representations in multi-view object class detection and scene understanding, their application in fine-grained recognition, an actively evolving domain within computer vision, has been notably scarce. Most leading approaches in fine-grained recognition still heavily rely on 2D image representations, which inherently limit their ability to capture intricate details, especially across various viewpoints. Understanding that the distinct characteristics defining fine-grained categories are more naturally represented in 3D object space, the authors aimed to rectify this gap. Their approach involved estimating the 3D geometry of objects to represent features in relation to this geometry, emphasizing both appearance and location of these features. Leveraging state-of-the-art 2D object representations and elevating them to 3D, the authors demonstrated the superiority of their 3D object representations in fine-grained categorization compared to existing 2D methods. Additionally, their contribution included introducing a new dataset encompassing 207 fine-grained categories, notably comprising a small-scale, ultra-fine-grained subset of 10 BMW models and a larger, more diverse set of 197 car types. The authors' work not only showcased the benefits of their 3D object representation in estimating 3D geometry but also explored the challenging task of 3D reconstruction for fine-grained categories, an area largely unexplored in existing literature.

## About Stanford Cars Dataset

Authors have collected a challenging, large-scale dataset of car models, to be made available upon publication. It consists of BMW-10, a small, ultra-fine-grained set of 10 BMW sedans (512 images) hand-collected by the authors, plus car-197, a large set of 197 car models (16,185 images) covering sedans, SUVs, coupes, convertibles, pickups, hatchbacks, and station wagons. Since dataset collection proved non-trivial, authors give the most important challenges and insights.

## Identifying visually distinct classes

Since cars are manmade objects whose class list changes on a yearly basis, and models of cars do not have a different appearance from year to year, no simple list of visually distinct cars exists which we can use as a base. We thus first crawl a popular car website for a list of all types of cars made since 1990. We then apply an aggressive deduplication procedure, based on perceptual hashing [35], to a limited number of provided example images for these classes, determining a subset of visually distinct classes, from which we sample 197 (see supplementary material for a complete list).

## Finding candidate images

Candidate images for each class were collected from Flickr, Google, and Bing. To reduce annotation cost and ensure diversity in the data, the candidate images for each class were deduplicated using the same perceptual hash algorithm [35], leaving a set of several thousand candidate images for each of the 197 target classes. These images were then put on Amazon Mechanical Turk (AMT) in order to determine whether they belong to their respective target classes.

## Training annotators

The main challenge in crowdsourcing the collection of a fine-grained dataset is that workers are typically non-experts. To compensate, we implemented a qualification task (a set of particularly hard examples of the actual annotation task) and provide a set of positive and negative example images for the car class a worker is annotating, drawing the negative examples from classes known a priori to be similar to the target class.

## Modeling annotator reliability

Even after training, workers differ in quality by large margins. To tackle this problem, we use the Get Another Label (GAL) system [15], which simultaneously estimates the probability a candidate image belongs to its target class and determines a quality level for each worker. Candidate images whose probability of belonging to the target class exceeds a specified threshold are then added to the set of images for that category. After obtaining images for each of the 197 target classes, we collect a bounding box for each image via AMT, using a quality-controlled system provided to us by the authors of [28]. Finally, an additional stage of deduplication is performed on the images when cropped to their bounding boxes.

<img width="903" alt="stanford_car_preview" src="https://github.com/dataset-ninja/stanford-cars/assets/123257559/6ceb3cb3-22f6-44d3-9fe2-c69d46e17afb">

<span style="font-size: smaller; font-style: italic;">One image each of 196 of the 197 classes in car-197 and each of the 10 classes in BMW-10.</span>
