import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random
import collections
from utils import clients
from agents.habitual_analyst import HabitualAnalyst
from agents.contextual_analyst import ContextualAnalyst
from agents.temporal_analyst import TemporalAnalyst
from agents.memory_master import MemoryMaster


IF_PROFILE = True  # Whether to use user profiles.

class MAPOI:
    """
    MAPOI: POI recommendation system based on a multi-agent framework.
    Uses pretrained weights and shared group knowledge to mitigate sparsity and cold-start issues.
    """
    def __init__(self):
        # self.random_seed = 42
        # Create agent instances.
        self.temporal_analyst = TemporalAnalyst(self)
        self.habitual_analyst = HabitualAnalyst(self)
        self.contextual_analyst = ContextualAnalyst(self)
        self.memory_master = MemoryMaster(mapoi_instance=self, memory_dir="./memory")
        self.customer_psychology_analyst = self.memory_master
        self.knowledge = {}  # Built or loaded lazily in run().
        self.transitions = {}

        if IF_PROFILE:
            self.profilename = "10_qwen7B"
            self.filename = self.profilename
        else:
            self.profilename = "10_qwen7B"
            self.filename = "noprofile"

        self.LLM_name = "re_qwen7B"  # LLM model name.
        print("MAPOI multi-agent POI recommender initialized.")

    def _load_transitions(self, dataset_name: str) -> dict:
        """
        Load transition priors from memory/{dataset}/transitions.json.
        Return an empty structure if missing to avoid downstream failures.
        """
        path = os.path.join("./memory", dataset_name.lower(), "transitions.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Fallback fields.
                data.setdefault("poi_to_poi", {})
                data.setdefault("cat_to_cat", {})
                return data
            except Exception as e:
                print(f"[MARPOI] Failed to read transitions: {e}")
        return {"poi_to_poi": {}, "cat_to_cat": {}}

    def run(self, data, datasetName, modelName):
        """
        Run the multi-agent POI recommendation system.
        
        Args:
            data: Tuple containing historical trajectories and related data.
            datasetName: Dataset name ('nyc' or 'tky').
            
        Returns:
            A list of evaluation metrics [acc@1, acc@10, mrr].
        """

        print(f"Starting MAPOI recommendation on dataset: {datasetName}...")
        
        # Build or load memory first (dynamic knowledge base).
        dataset_base = os.path.join("data", datasetName.lower())
        input_candidates = [
            os.path.join(dataset_base, "new_train_sample.csv"),
            os.path.join(dataset_base, "new_test_sample.csv"),
        ]
        self.knowledge = self.memory_master.build_or_load_memory(
            dataset_name=datasetName,
            input_paths=input_candidates,
            force_rebuild=False,
            save_transitions=True
        )
        # 2) Load transitions (same lifecycle as knowledge).
        self.transitions = self._load_transitions(datasetName)
        print(
            f"[MARPOI] transitions loaded: poi_to_poi={len(self.transitions.get('poi_to_poi', {}))} source nodes, "
            f"cat_to_cat={len(self.transitions.get('cat_to_cat', {}))} source categories"
        )

        # Unpack data.
        longs, recents, targets, poiInfos, traj2u = data

        # Initialize statistics and locks.
        self.hit1 = 0
        self.hit5 = 0
        self.rr = 0
        self.err = []
        self.stats_lock = threading.Lock()
        
        # For valid recommendation statistics.
        self.legal_counts = []
        self.legal_counts_lock = threading.Lock()
        # For habit_confidence score statistics.
        self.habit_confidence_scores = []
        self.habit_confidence_lock = threading.Lock()

        # Store shared data for agents.
        self.longs = longs
        self.recents = recents 
        self.poiInfos = poiInfos
        self.traj2u = traj2u
        self.datasetName = datasetName
        self.modelName = modelName

        # Prepare candidate sets and test tasks.
        poiList = list(poiInfos.keys())
        tasks = []
        
        for trajectory, groundTruth in targets.items():
            seed_value = eval(trajectory)
            random.seed(seed_value)
            negSample = random.sample(poiList, 100)
            candidateSet = negSample + [groundTruth[0]]
            tasks.append((trajectory, candidateSet, groundTruth))
        
        # Process trajectories in parallel with a thread pool.
        with ThreadPoolExecutor(max_workers=len(clients)*4) as executor:
            list(tqdm(executor.map(lambda x: self._process_trajectory(*x), tasks), total=len(tasks)))
        
        # Process error list.
        self.err = list(sorted(self.err))
        with open('./testERR', 'w') as file:
            file.write(str(self.err))
        
        # Compute evaluation metrics.
        # Count trajectories with successful output.

        num_trajectories = len(targets)
        acc1 = self.hit1 / num_trajectories
        acc5 = self.hit5 / num_trajectories
        # Compute mean reciprocal rank (MRR).
        mrr = self.rr / num_trajectories
        
        # --- Additional statistics: valid-count distribution and extreme trajectory IDs ---
        if self.legal_counts:
            min_legal = min(self.legal_counts)
            max_legal = max(self.legal_counts)
            avg_legal = sum(self.legal_counts) / len(self.legal_counts)
            print(f'Valid recommendation count in candidate set: min={min_legal}, max={max_legal}, avg={avg_legal:.2f}')

            # Histogram (distribution statistics).
            histogram = collections.Counter(self.legal_counts)
            print("Valid-count histogram (valid count: frequency):")
            for legal_num in range(min_legal, max_legal + 1):
                print(f"{legal_num}: {histogram.get(legal_num, 0)}")

            # Print trajectory IDs for edge valid-count cases.
            zero_legal_ids = [tid for tid, n in self.legal_count_id_list if n == 0]
            max_legal_ids = [tid for tid, n in self.legal_count_id_list if n == max_legal]
            print(f'Trajectory IDs with valid count = 0: {zero_legal_ids}')
            print(f'Trajectory IDs with valid count = {max_legal}: {max_legal_ids}')
        else:
            print('No valid recommendation statistics available.')

        print(f'acc@1: {acc1}, acc@5: {acc5}, mrr@5: {mrr}')

        if self.habit_confidence_scores:
            avg_confidence = sum(self.habit_confidence_scores) / len(self.habit_confidence_scores)
            print(f'Average habit confidence score: {avg_confidence:.4f}')
            # Print habit confidence score histogram.
            conf_histogram = collections.Counter()
            for score in self.habit_confidence_scores:
                bucket = round(score, 2)  # Bucket by 0.01.
                conf_histogram[bucket] += 1
            print("Habit confidence score histogram (score: frequency):")
            for bucket in sorted(conf_histogram.keys()):
                print(f"{bucket}: {conf_histogram[bucket]}")
        
        return acc1, acc5, mrr  # Metrics in requested order [acc@1, acc@5, mrr].

    def _process_trajectory(self, trajectory, candidateSet, groundTruth):
        """Process one trajectory for thread-pool execution."""
        try:
            prediction, habit_confidence = self.runeach(trajectory, candidateSet, groundTruth)

            # Valid recommendation statistics.
            legal_num = sum([poi in candidateSet for poi in prediction])
            with self.legal_counts_lock:
                self.legal_counts.append(legal_num)
                # Record trajectory ID and valid count.
                if not hasattr(self, 'legal_count_id_list'):
                    self.legal_count_id_list = []
                self.legal_count_id_list.append((trajectory, legal_num))
            # habit_confidence_score statistics.
            with self.habit_confidence_lock:
                self.habit_confidence_scores.append(habit_confidence)

            with self.stats_lock:
                if groundTruth[0] in prediction:
                    index = prediction.index(groundTruth[0]) + 1
                    if index == 1:
                        self.hit1 += 1
                    if index <= 5:
                        self.hit5 += 1
                    self.rr += 1 / index
                else:
                    self.err.append(eval(trajectory))
        except Exception as e:
            print(f"Error while processing trajectory {trajectory}: {repr(e)}")
            with self.stats_lock:
                self.err.append(eval(trajectory))
    
    def runeach(self, trajectory, candidateSet, groundTruth):
        """
        Run multi-agent recommendation for a single trajectory.
        """
        # 1. Memory analyst performs data preprocessing and trajectory enhancement.
        data = self.customer_psychology_analyst.data_preprocessing(trajectory, candidateSet, groundTruth)
        user_id, longterm_groups, longterm, recent, next_time_day, next_time_period, candidates, current_poi, current_time = data

        # 2. Memory analyst generates user profile.
        profile_path = f'./memory/{self.datasetName}/profile/{trajectory}'
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as file:
                profile_content = json.load(file)
                # Use existing user profile.
                profile = profile_content["profile"]
        
        else:
            # print(f"Regenerating profile for user {user_id}...")
            output_profile = {}
            output_profile["user_id"] = user_id
            profile_prompt, profile = self.customer_psychology_analyst.user_profilling(
                user_id, longterm_groups, recent, next_time_day, next_time_period
            )
            # Save profile-related information.
            output_profile["profile_prompt"] = profile_prompt
            output_profile["profile"] = profile
            # Save result.
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            with open(profile_path, 'w') as file:
                file.write(json.dumps(output_profile, indent='\t'))

        # 3. Recommendation specialists generate parallel recommendations.
        agent_data = {
            "user_id": user_id,
            "profile": profile, 
            "long": longterm,
            "recent": recent,
            "candidates": candidates,
            "current_time": current_time,
            "current_poi": current_poi,
            "next_time_day": next_time_day,
            "next_time_period": next_time_period,
        }
        candidate_pois = [poi for poi, _, _ in candidates]
        path4 = f'./output_pkdd/HabitualAnalyst/{self.datasetName}_{self.filename}_{self.LLM_name}/{trajectory}'
        if os.path.exists(path4):
            with open(path4, 'r') as file:
                res_content = json.load(file)
                agent_result4 = res_content["response"]
                candidates = agent_result4["recommendation"]
                habit_confidence = agent_result4["confidence"]
                candidates = [poi for poi in candidates if poi in candidate_pois]
        else:
            agent_result4, candidates = self.habitual_analyst.generate_recommendation(agent_data, candidate_pois, IF_PROFILE)
            output4 = {"user_id": user_id, "trajectory": trajectory, "response": agent_result4, "groundTruth": groundTruth[0]}
            os.makedirs(os.path.dirname(path4), exist_ok=True)
            with open(path4, 'w') as file:
                file.write(json.dumps(output4, indent='\t'))
        un_candidates = [poi for poi in candidate_pois if poi not in candidates]
        candidates = candidates + un_candidates
        agent_data["candidates"] = [(poi, self.poiInfos[poi]["category"]) for poi in candidates[:20]]
        final_result = {"recommendation": candidates[:20]}
        habit_confidence = agent_result4["confidence"]

        # If habit_confidence is at or below threshold, use additional agents.
        threshold = 0.9
        if habit_confidence < threshold or habit_confidence == threshold:
            path5 = f'./output_pkdd/ContextualAnalyst/{self.datasetName}_{self.filename}_{self.LLM_name}/{trajectory}'
            if os.path.exists(path5):
                with open(path5, 'r') as file:
                    res_content = json.load(file)
                    agent_result5 = res_content["response"]
                    candidates = agent_result5["recommendation"]
                    candidates = [poi for poi in candidates if poi in candidate_pois]
            else:
                agent_result5, candidates = self.contextual_analyst.generate_recommendation(agent_data, candidate_pois, IF_PROFILE)
                output5 = {"user_id": user_id, "trajectory": trajectory, "response": agent_result5, "groundTruth": groundTruth[0]}
                os.makedirs(os.path.dirname(path5), exist_ok=True)
                with open(path5, 'w') as file:
                    file.write(json.dumps(output5, indent='\t'))
            agent_data["candidates"] = [(poi, self.poiInfos[poi]["category"]) for poi in candidates[:20]]
            final_result = {"recommendation": candidates[:20]}

            path2 = f'./output_pkdd/TemporalAnalyst/{self.datasetName}_{self.filename}_{self.LLM_name}/{trajectory}'
            if os.path.exists(path2):
                with open(path2, 'r') as file:
                    res_content = json.load(file)
                    agent_result2 = res_content["response"]
                    candidates = agent_result2["recommendation"]
                    candidates = [poi for poi in candidates if poi in candidate_pois]
            else:
                agent_result2, candidates = self.temporal_analyst.generate_recommendation(agent_data, candidate_pois, IF_PROFILE)
                output2 = {"user_id": user_id, "trajectory": trajectory, "response": agent_result2, "groundTruth": groundTruth[0]}
                os.makedirs(os.path.dirname(path2), exist_ok=True)
                with open(path2, 'w') as file:
                    file.write(json.dumps(output2, indent='\t'))
            agent_data["candidates"] = [(poi, self.poiInfos[poi]["category"]) for poi in candidates[:20]]
            final_result = {"recommendation": candidates[:20]}

            path1= f'./output_pkdd/HabitualAnalyst_again/{self.datasetName}_{self.filename}_{self.LLM_name}/{trajectory}'
            if os.path.exists(path1):
                with open(path1, 'r') as file:
                    res_content = json.load(file)
                    agent_result4 = res_content["response"]
                    candidates = agent_result4["recommendation"]
                    candidates = [poi for poi in candidates if poi in candidate_pois]
            else:
                agent_result1, candidates = self.habitual_analyst.generate_recommendation(agent_data, candidate_pois, IF_PROFILE)
                output1 = {"user_id": user_id, "trajectory": trajectory, "response": agent_result1, "groundTruth": groundTruth[0]}
                os.makedirs(os.path.dirname(path1), exist_ok=True)
                with open(path1, 'w') as file:
                    file.write(json.dumps(output1, indent='\t'))
            un_candidates = [poi for poi in candidate_pois if poi not in candidates]
            candidates = candidates + un_candidates
            agent_data["candidates"] = [(poi, self.poiInfos[poi]["category"]) for poi in candidates[:20]]
            final_result = {"recommendation": candidates[:20]}

        # Save output result.
        output = {}
        output["user_id"] = user_id
        output["trajectory"] = trajectory
        output["response"] = final_result
        output["groundTruth"] = groundTruth[0]
        
        # Save result.
        self._save_output(output, trajectory)

        return final_result["recommendation"][:5], habit_confidence

    def _save_output(self, output, trajectory):
        """Save recommendation result to file."""
        path = f'./output/{self.modelName}/{self.datasetName}_{self.filename}_{self.LLM_name}/{trajectory}'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            file.write(json.dumps(output, indent='\t'))
