# Train Test Split

Testing the balanced split concept from the [Balanced Split](https://arxiv.org/abs/2212.11116) paper.

## Paper Summary

- Addressed the idea of imbalanced datasets and suggested training with balanced data extracted from the originally unbalanced.
- Introduced a restriction on the size of the training data to be less than or equal to the minority class ratio multiplied by the total number of classes.
- Provided comparison between the pros and cons of random, stratified, and balanced splitting methods.
- Showed improved performance over the other two methods.

## Split Function

The following is the split function used to prove the concept.

```python
def split(ds, split="stratified", seed=42, train_size=0.75):
    splits = None
    if split == "stratified":
        splits = train_test_split(
            ds, stratify=ds.labels, random_state=seed, train_size=train_size
        )
    elif split == "balanced":
        class_ratios = ds.labels.value_counts(normalize=True)
        classes = ds.labels.unique()
        num_classes = len(classes)
        min_ratio = min(class_ratios.to_list())
        train_size = min(train_size, num_classes * min_ratio)
        print(f"Train size used: {train_size}")
        class_ratio = train_size / num_classes
        examples_per_class = int(class_ratio * len(ds))

        inds = []
        for c in classes:
            sample = ds[ds.labels == c].sample(examples_per_class, random_state=seed)
            inds.extend(sample.index.to_list())
        splits = (ds.iloc[inds, :], ds.drop(index=inds))
    else:
        raise Exception("Unknown split method")
    return splits
```

## Dataset

The concept was test using the [20-news-groups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) dataset from Sci-Kit Learn.

## Baseline Result

The base line model used is **Logistic Regression**.

Three configurations were test:

- Balanced Split
- Stratified Split
- Stratified Split (Weighted Learning)

The results came as follows.

|   Score \ Method   | Balanced | Stratified | Stratified (Weighted) |
| :----------------: | :------: | :--------: | :-------------------: |
| F1 (weighted avg.) |   0.68   |    0.69    |         0.67          |

## Transformer Model

To verify the idea on a more realistic model, a [Bert-Base-Cased](https://huggingface.co/bert-base-cased) model was used as a sequence classifier.

The results came as follows.

|   Score \ Method   | Balanced | Stratified | Stratified (Weighted) |
| :----------------: | :------: | :--------: | :-------------------: |
| F1 (weighted avg.) |  0.785   |   0.775    |         0.783         |

## Conclusion

It is hard to tell but the method is worth testing whenever possible. We still need a robust method to split our datasets properly based on the input features and labels at the same time. For example, we need to have the same distribution of labels in training, validation, and testing. However, we also need to include all types of rules why an example is give a speciific label at least in the training split. To elaborate, an example may be positive (sentiment) based on using positive words or negating negative ones. We need to ensure that these two signals are present at least in the training split.
