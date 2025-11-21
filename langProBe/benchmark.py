import random, os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Type

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import Teleprompter

import langProBe.optimizers as langprobe_optimizers
from langProBe.dspy_program import LangProBeDSPyMetaProgram
from langProBe.config_utils import read_json, read_jsonl
from langProBe.program_utils import ProcessManager




dataset_size = {"full": None, "lite": 500, "tiny": 200, "test": 2}


class Benchmark(ABC):
    def __init__(self, dataset_mode="lite"):
        # dataset for training and validation
        self.dataset = None
        # dataset for the actual benchmarking
        self.test_set = None
        self.train_set = None
        self.dev_set = None
        self.val_set = None

        self.init_dataset()
        assert self.dataset is not None, "Dataset not initialized"
        assert self.test_set is not None, "Test set not initialized"
        self.max_testset_size = dataset_size[dataset_mode]

        self.test_set = self.trim_dataset(self.test_set, self.max_testset_size)

        # TODO: FIXME: "test" option is for debugging purposes only, should be removed for final release
        if dataset_mode == "test":
            self.dataset = self.trim_dataset(self.dataset, 60)
            self.create_splits()
            self.test_set = self.trim_dataset(self.test_set, 50)

        if not self.train_set or not self.dev_set or not self.val_set:
            self.create_splits()

        self.train_set = self.trim_dataset(self.train_set, 150)
        self.dev_set = self.trim_dataset(self.dev_set, 300)
        self.val_set = self.trim_dataset(self.val_set, 300)

        assert self.train_set is not None, "Train set not initialized"
        assert self.dev_set is not None, "Dev set not initialized"
        assert self.val_set is not None, "Val set not initialized"

    @abstractmethod
    def init_dataset(self) -> None:
        """
        Initializes the dataset for the benchmark, and sets it to self.dataset.
        Each element in the dataset should be an instance of dspy.Example.
        """
        return

    def trim_dataset(self, dataset, size: int) -> None:
        if size is None or size >= len(dataset):
            return dataset
        rng = random.Random()
        rng.seed(1)
        return rng.sample(dataset, size)

    def create_splits(self) -> None:
        """
        Creates the splits for the dataset (not including test).
        Upon completion, self.train_set, self.dev_set, and self.val_set should be set.
        """

        total_len = len(self.dataset)
        self.dev_set = self.dataset[: int(0.4 * total_len)]
        self.val_set = self.dataset[int(0.4 * total_len) : int(0.8 * total_len)]
        self.train_set = self.dataset[int(0.8 * total_len) :]

    def get_dataset(self):
        return self.dataset

    def get_train_set(self):
        return self.train_set

    def get_dev_set(self):
        return self.dev_set

    def get_test_set(self):
        return self.test_set


class MCPBench(Benchmark):
    def __init__(self, dataset_mode="lite", dataset_path=None, missing_data=[]):
        self.dataset_path = dataset_path
        self.missing_data = missing_data
        super().__init__(dataset_mode=dataset_mode)

    def init_dataset(self):
        self.dataset = []
        self.test_set = []
        if self.missing_data:
            test_raw_data = self.missing_data
        else:
            test_raw_data = read_jsonl(self.dataset_path)
        
        for test_data in test_raw_data:
            self.test_set.append(
                dspy.Example(
                    id=test_data["unique_id"],
                    question=test_data["Prompt"],
                    answer=test_data["Answer"],
                ).with_inputs("id", "question", "answer", "config")
            )




@dataclass
class EvaluationResult:
    benchmark: str
    program: str

    score: float
    cost: float
    input_tokens: int
    output_tokens: int

    outputs_raw_data: List|None = None

    # optimizer: str = None
    # optimized_program: dspy.Module = None
    # optimizer_input_tokens: int = None
    # optimizer_output_tokens: int = None
    # optimizer_cost: float = None

    # optimizer_program_scores: list[float] = None


@dataclass
class BenchmarkMeta:
    benchmark: Type[Benchmark]
    program: List[dspy.Module]
    metric: Callable
    dataset_mode: str = "lite"

    optimizers: List[langprobe_optimizers.OptimizerConfig] = field(
        default_factory=lambda: langprobe_optimizers.DEFAULT_OPTIMIZERS
    )

    # BenchmarkMeta.num_threads has higher priority than run time argument of num_threads
    # use this as an upper bound for the number of threads to use
    num_threads: int = None
    name: str = None


def setup_lm(dspy_config=None):
    lm: dspy.LM = dspy_config.get("lm", dspy.settings.lm)
    assert lm is not None, "dspy language model not set"

    lm = lm.copy()
    assert len(lm.history) == 0, "language model history not empty"
    return lm


# def calculate_stats(lm: dspy.LM) -> tuple[float, int, int]:
#     cost = 0
#     input_tokens = 0
#     output_tokens = 0
#     for i, trace in enumerate(lm.history):
#         cost += trace.get("cost", None) or 0
#         input_tokens += trace.get("usage", 0).get("prompt_tokens", 0)
#         output_tokens += trace.get("usage", 0).get("completion_tokens", 0)

#     return cost, input_tokens, output_tokens

def calculate_stats(manager: List[ProcessManager]) -> tuple[float, float, float]:
    input_tokens = sum(usage["prompt_tokens"] for trace in manager for usage in trace.lm_usages)
    output_tokens = sum(usage["completion_tokens"] for trace in manager for usage in trace.lm_usages)
    
    avg_input = input_tokens // len(manager)
    avg_output = output_tokens // len(manager)
    
    return 0, avg_input, avg_output



class EvaluateBench(ABC):
    def __init__(
        self,
        benchmark: Benchmark,
        program: dspy.Module,
        metric: Callable,
        lm: str,
        benchmark_name: str = None,
        num_threads: int = 1,
        api_key: str = None,
        api_base: str = None,
    ):
        self.benchmark = benchmark
        self.program = program

        self.program.setup_lm(lm, api_key=api_key, api_base=api_base)
        self.metric = metric
        self.num_threads = num_threads
        devset = benchmark.get_test_set()
        self.evaluate_prog = Evaluate(
            devset=devset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_progress=True,
            max_errors=5000,
            # return_outputs=True,
            provide_traceback=True,
        )

        self.program_name = getattr(
            self.program, "_name", self.program.__class__.__name__
        )
        self.benchmark_name = benchmark_name or self.benchmark.__class__.__name__
        self.results: list[EvaluationResult] = []

    def get_empty_results(self):
        return EvaluationResult(
            benchmark=self.benchmark_name,
            program=self.program_name,
            score=0,
            cost=0,
            input_tokens=0,
            output_tokens=0,
        )


    def evaluate_baseline(self, dspy_config=None) -> EvaluationResult:
        with dspy.context(**dspy_config):
            eval_result = self.evaluate_prog(self.program)
        result = self.get_empty_results()
        
        # eval_result is an EvaluationResult object with:
        # - score: float (percentage score)
        # - results: list of (example, prediction, score) tuples
        score = eval_result.score
        results = eval_result.results
        
        # Unpack results: each element is (example, prediction, score)
        datasets, outputs, _ = zip(*results)
        managers = [one.process_report for one in outputs]

        result.score = score   
        result.outputs_raw_data = outputs
        result.cost, result.input_tokens, result.output_tokens = calculate_stats(managers)

        return result

    def evaluate(self, dspy_config=None) -> EvaluationResult:
        """
        Args:
            dspy_config: A dictionary of configurations for dspy.context
        Returns:
            A list of EvaluationResult objects.
        """
        if dspy_config is None:
            dspy_config = {}

        result = self.evaluate_baseline(dspy_config)
        self.results = result
        return result
