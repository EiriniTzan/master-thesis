"""Small CapyMOA utilities for STUDD experiments."""

from capymoa.classifier import (
    AdaptiveRandomForestClassifier,
    HoeffdingTree,
    StreamingGradientBoostedTrees,
    NaiveBayes,
    SGDClassifier
)


def make_rf(schema, seed: int, n_trees: int):
    """
    Create an Adaptive Random Forest classifier.

    The forest is configured without internal drift detection and
    background learners because drift monitoring is performed
    externally by STUDD.

    Parameters
    ----------
    schema : Schema
        Stream schema obtained from the CapyMOA dataset.
    seed : int
        Random seed used for reproducibility.
    n_trees : int
        Number of trees in the ensemble.

    Returns
    -------
    AdaptiveRandomForestClassifier
        Configured Adaptive Random Forest classifier.
    """

    return AdaptiveRandomForestClassifier(
        schema=schema,
        ensemble_size=n_trees,
        random_seed=seed,
        disable_drift_detection=True,
        disable_background_learner=True,
    )


def make_teacher(schema, teacher_type: str, seed: int, n_trees: int):
    """
    Create the teacher model used by STUDD.

    Supported teacher models are:
    - Random Forest ("rf")
    - Naive Bayes ("nb")
    - Logistic Regression implemented through SGDClassifier ("lr")
    - Streaming Gradient Boosted Trees ("sgbt")

    Parameters
    ----------
    schema : Schema
        Stream schema obtained from the dataset.
    teacher_type : str
        Type of teacher model ("rf", "nb", or "lr").
    seed : int
        Random seed used for reproducibility.
    n_trees : int
        Number of trees when using Random Forest.

    Returns
    -------
    Classifier
        Initialized CapyMOA classifier.

    Raises
    ------
    ValueError
        If an unsupported teacher type is requested.
    """

    if teacher_type == "rf":
        return make_rf(schema=schema, seed=seed, n_trees=n_trees)

    if teacher_type == "nb":
        return NaiveBayes(schema=schema, random_seed=seed)
    
    if teacher_type == "lr":
        return SGDClassifier(schema=schema, loss="log_loss", random_seed=seed)
    
    if teacher_type == "sgbt":
        return StreamingGradientBoostedTrees(schema=schema)

    raise ValueError(f"Unknown teacher_type={teacher_type!r}. Use 'rf', 'lr', 'nb' or 'sgbt'.")


def make_student(schema, student_type: str, seed: int, n_trees: int):
    """
    Create the student model used by STUDD.

    Supported student models are:
    - Random Forest ("rf")
    - Hoeffding Tree ("ht")
    - Streaming Gradient Boosted Trees ("sgbt")
    - Logistic Regression via SGDClassifier ("lr")

    Parameters
    ----------
    schema : Schema
        Stream schema obtained from the dataset.
    student_type : str
        Type of student model.
    seed : int
        Random seed used for reproducibility.
    n_trees : int
        Number of trees when using Random Forest.

    Returns
    -------
    Classifier
        Initialized CapyMOA classifier.

    Raises
    ------
    ValueError
        If an unsupported student type is requested.
    """

    if student_type == "rf":
        return make_rf(schema=schema, seed=seed, n_trees=n_trees)

    if student_type == "ht":
        return HoeffdingTree(schema=schema)

    if student_type == "sgbt":
        return StreamingGradientBoostedTrees(schema=schema)
    
    if student_type == "lr":
        return SGDClassifier(schema=schema, loss="log_loss", random_seed=seed)

    raise ValueError(f"Unknown student_type={student_type!r}. Use 'rf', 'ht', 'lr' or 'sgbt'.")


def collect_instances(stream, n: int):
    """
    Collect a fixed number of instances from a stream.

    The function iteratively reads instances from a CapyMOA stream
    until either `n` instances have been collected or the stream
    is exhausted.

    Parameters
    ----------
    stream : Stream
        CapyMOA stream object.
    n : int
        Maximum number of instances to collect.

    Returns
    -------
    list
        List containing the collected stream instances.
    """
    
    instances = []

    for _ in range(n):
        if not stream.has_more_instances():
            break
        instances.append(stream.next_instance())

    return instances


