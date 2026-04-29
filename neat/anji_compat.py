"""ANJI compatibility mode for neat-python.

This module provides ANJI-inspired implementations that can be selected via
``algorithm_mode = anji`` in the ``[NEAT]`` config section.
"""

import math
import random
from itertools import count

from neat.config import ConfigParameter, DefaultClassConfig
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.innovation import InnovationTracker
from neat.math_util import mean
from neat.species import DefaultSpeciesSet
from neat.stagnation import stat_functions


class AnjiInnovationTracker(InnovationTracker):
    """Innovation tracker with ANJI-style persistent mappings."""

    def __init__(self, start_number=0):
        super().__init__(start_number=start_number)
        self._split_connection_to_node = {}

    def reset_generation(self):
        """No-op: ANJI mappings are persistent, not generation-scoped."""
        return

    def get_or_create_split_node(self, split_connection_innovation, create_node_id_fn):
        node_id = self._split_connection_to_node.get(split_connection_innovation)
        if node_id is None:
            node_id = create_node_id_fn()
            self._split_connection_to_node[split_connection_innovation] = node_id
        return node_id


class AnjiGenome(DefaultGenome):
    """DefaultGenome with ANJI-inspired mutation semantics."""

    config_section_name = "DefaultGenome"

    @classmethod
    def parse_config(cls, param_dict):
        param_dict["node_gene_type"] = cls.parse_node_gene_type(param_dict)
        param_dict["connection_gene_type"] = cls.parse_connection_gene_type(param_dict)
        cfg = DefaultGenomeConfig(param_dict, "DefaultGenome")

        # ANJI compatibility knobs (optional; defaults preserve ANJI-like behavior).
        cfg.anji_topology_mutation_classic = str(
            param_dict.get("anji_topology_mutation_classic", "false")
        ).lower() in ("1", "true", "yes", "on")
        cfg.anji_mutate_nodes = str(
            param_dict.get("anji_mutate_nodes", "false")
        ).lower() in ("1", "true", "yes", "on")
        cfg.anji_prune = str(param_dict.get("anji_prune", "true")).lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        cfg.anji_prune_rate = float(param_dict.get("anji_prune_rate", 1.0))
        cfg.anji_remove_connection_rate = float(
            param_dict.get("anji_remove_connection_rate", cfg.conn_delete_prob)
        )
        return cfg

    @staticmethod
    def parse_node_gene_type(param_dict):
        # Keep default gene types unless explicitly injected by callers.
        from neat.genes import DefaultNodeGene

        return param_dict.get("node_gene_type", DefaultNodeGene)

    @staticmethod
    def parse_connection_gene_type(param_dict):
        from neat.genes import DefaultConnectionGene

        return param_dict.get("connection_gene_type", DefaultConnectionGene)

    def mutate(self, config):
        if getattr(config, "anji_topology_mutation_classic", False):
            add_conn = max(0.0, float(config.conn_add_prob))
            add_node = max(0.0, float(config.node_add_prob))
            p_any = add_conn + add_node - (add_conn * add_node)
            if random.random() < p_any:
                if (add_conn + add_node) > 0.0 and random.random() < (add_conn / (add_conn + add_node)):
                    self._mutate_add_connection_opportunity(config, single=True)
                else:
                    self._mutate_add_node_opportunity(config, single=True)
        else:
            self._mutate_add_connection_opportunity(config, single=False)
            self._mutate_add_node_opportunity(config, single=False)

        self._mutate_remove_connection_opportunity(config)

        # ANJI mutates connection weights; node genes are generally fixed.
        for cg in self.connections.values():
            cg.mutate(config)
        if getattr(config, "anji_mutate_nodes", False):
            for ng in self.nodes.values():
                ng.mutate(config)

        if getattr(config, "anji_prune", True):
            prune_rate = max(0.0, min(1.0, float(getattr(config, "anji_prune_rate", 1.0))))
            if random.random() < prune_rate:
                self._prune_anji_stranded_alleles(config)

    def _candidate_connection_keys(self, config):
        possible_outputs = list(self.nodes)
        possible_inputs = list((set(self.nodes) - set(config.output_keys)) | set(config.input_keys))

        candidates = []
        for in_node in possible_inputs:
            for out_node in possible_outputs:
                key = (in_node, out_node)
                if key in self.connections:
                    continue
                if in_node in config.output_keys and out_node in config.output_keys:
                    continue
                if config.feed_forward and self._would_create_cycle(key):
                    continue
                candidates.append(key)
        return candidates

    def _would_create_cycle(self, key):
        from neat.graphs import creates_cycle

        return creates_cycle(list(self.connections), key)

    def _mutate_add_connection_opportunity(self, config, single=False):
        assert config.innovation_tracker is not None, "Innovation tracker must be set"
        candidates = self._candidate_connection_keys(config)
        if not candidates:
            return

        random.shuffle(candidates)
        if single:
            in_node, out_node = candidates[0]
            innovation = config.innovation_tracker.get_innovation_number(
                in_node, out_node, "add_connection"
            )
            cg = self.create_connection(config, in_node, out_node, innovation)
            self.connections[cg.key] = cg
            return

        for in_node, out_node in candidates:
            if random.random() < config.conn_add_prob:
                innovation = config.innovation_tracker.get_innovation_number(
                    in_node, out_node, "add_connection"
                )
                cg = self.create_connection(config, in_node, out_node, innovation)
                self.connections[cg.key] = cg

    def _mutate_add_node_opportunity(self, config, single=False):
        assert config.innovation_tracker is not None, "Innovation tracker must be set"
        connections = list(self.connections.values())
        if not connections:
            return
        random.shuffle(connections)

        if single:
            self._split_connection_anji(connections[0], config)
            return

        for conn in connections:
            if random.random() < config.node_add_prob:
                self._split_connection_anji(conn, config)

    def _split_connection_anji(self, conn_to_split, config):
        tracker = config.innovation_tracker
        split_innovation = conn_to_split.innovation

        if hasattr(tracker, "get_or_create_split_node"):
            new_node_id = tracker.get_or_create_split_node(
                split_innovation, lambda: config.get_new_node_key(self.nodes)
            )
        else:
            new_node_id = config.get_new_node_key(self.nodes)

        if new_node_id not in self.nodes:
            ng = self.create_node(config, new_node_id)
            if hasattr(ng, "bias"):
                ng.bias = 0.0
            self.nodes[new_node_id] = ng

        conn_to_split.enabled = False
        i, o = conn_to_split.key
        in_innov = tracker.get_innovation_number(i, new_node_id, "add_connection")
        out_innov = tracker.get_innovation_number(new_node_id, o, "add_connection")
        self.add_connection(config, i, new_node_id, 1.0, True, innovation=in_innov)
        self.add_connection(
            config, new_node_id, o, conn_to_split.weight, True, innovation=out_innov
        )

    def _mutate_remove_connection_opportunity(self, config):
        if not self.connections:
            return
        rate = max(0.0, float(getattr(config, "anji_remove_connection_rate", config.conn_delete_prob)))
        if rate <= 0.0:
            return
        to_delete = []
        for key in list(self.connections.keys()):
            if random.random() < rate:
                to_delete.append(key)
        for key in to_delete:
            self.connections.pop(key, None)

    def _find_anji_unvisited(self, config, is_forward):
        """Return (unvisited_hidden_nodes, unvisited_enabled_connection_keys)."""
        enabled_connection_keys = {
            cg.key for cg in self.connections.values() if cg.enabled
        }
        unvisited_connections = set(enabled_connection_keys)

        output_keys = set(config.output_keys)
        input_keys = set(config.input_keys)
        hidden_nodes = set(self.nodes.keys()) - output_keys - input_keys
        unvisited_hidden_nodes = set(hidden_nodes)

        current_nodes = set(config.input_keys if is_forward else config.output_keys)
        next_nodes = set()
        while unvisited_connections and current_nodes:
            next_nodes.clear()
            visited_now = []
            for src, dst in list(unvisited_connections):
                if (is_forward and src in current_nodes) or ((not is_forward) and dst in current_nodes):
                    visited_now.append((src, dst))
                    next_nodes.add(dst if is_forward else src)

            if not visited_now:
                break

            for key in visited_now:
                unvisited_connections.discard(key)

            unvisited_hidden_nodes -= next_nodes
            current_nodes = set(next_nodes)

        return unvisited_hidden_nodes, unvisited_connections

    def _prune_anji_stranded_alleles(self, config):
        """ANJI-style prune: remove nodes/connections stranded in either direction."""
        fwd_unvisited_nodes, fwd_unvisited_connections = self._find_anji_unvisited(
            config, is_forward=True
        )
        rev_unvisited_nodes, rev_unvisited_connections = self._find_anji_unvisited(
            config, is_forward=False
        )

        stranded_nodes = set(fwd_unvisited_nodes) | set(rev_unvisited_nodes)
        stranded_connections = (
            set(fwd_unvisited_connections) | set(rev_unvisited_connections)
        )

        if not stranded_nodes and not stranded_connections:
            return

        for node_key in stranded_nodes:
            self.nodes.pop(node_key, None)

        for conn_key in list(self.connections.keys()):
            src, dst = conn_key
            if conn_key in stranded_connections or src in stranded_nodes or dst in stranded_nodes:
                self.connections.pop(conn_key, None)


class AnjiSpeciesSet(DefaultSpeciesSet):
    """Species set with static threshold behavior (ANJI-style)."""

    config_section_name = "DefaultSpeciesSet"

    @classmethod
    def parse_config(cls, param_dict):
        cfg = DefaultSpeciesSet.parse_config(param_dict)
        cfg.target_num_species = "none"
        return cfg

    def speciate(self, config, population, generation):
        # Force static threshold behavior regardless of dynamic knobs.
        self.species_set_config.target_num_species = "none"
        super().speciate(config, population, generation)


class AnjiNoStagnation(DefaultClassConfig):
    """ANJI-style stagnation policy: no species are explicitly removed for stagnation."""

    config_section_name = "DefaultStagnation"

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("species_fitness_func", str, "mean"),
                ConfigParameter("max_stagnation", int, 15),
                ConfigParameter("species_elitism", int, 0),
            ],
            "DefaultStagnation",
        )

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.stagnation_config = config
        self.species_fitness_func = stat_functions.get(config.species_fitness_func)
        if self.species_fitness_func is None:
            raise RuntimeError(
                f"Unexpected species fitness func: {config.species_fitness_func!r}"
            )
        self.reporters = reporters

    def update(self, species_set, generation):
        result = []
        for sid in sorted(species_set.species.keys()):
            s = species_set.species[sid]
            s.fitness = self.species_fitness_func(s.get_fitnesses())
            s.fitness_history.append(s.fitness)
            if not s.fitness_history or s.fitness >= max(s.fitness_history):
                s.last_improved = generation
            s.adjusted_fitness = None
            result.append((sid, s, False))
        return result


class AnjiReproduction(DefaultClassConfig):
    """ANJI-style reproduction with survivor selection + clone/crossover slices."""

    config_section_name = "DefaultReproduction"

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                # Keep default neat-python keys for compatibility with existing configs.
                ConfigParameter("elitism", int, 0),
                ConfigParameter("survival_threshold", float, 0.2),
                ConfigParameter("min_species_size", int, 1),
                ConfigParameter("fitness_sharing", str, "canonical"),
                ConfigParameter("spawn_method", str, "proportional"),
                ConfigParameter("interspecies_crossover_prob", float, 0.0),
                # ANJI-like keys.
                ConfigParameter("anji_survival_rate", float, 0.2),
                ConfigParameter("anji_elitism", bool, True),
                ConfigParameter("anji_elitism_min_species_size", int, 6),
                ConfigParameter("anji_clone_slice", str, "auto"),
                ConfigParameter("anji_crossover_slice", str, "auto"),
            ],
            "DefaultReproduction",
        )

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}
        self.innovation_tracker = AnjiInnovationTracker()

    def create_new(self, genome_type, genome_config, num_genomes):
        genome_config.innovation_tracker = self.innovation_tracker
        new_genomes = {}
        for _ in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()
        return new_genomes

    def _species_fitness(self, specie):
        member_fitnesses = [m.fitness for m in specie.members.values()]
        if not member_fitnesses:
            return 0.0
        return mean(member_fitnesses)

    def _select_survivors(self, species_list, pop_size):
        survival_rate = self.reproduction_config.anji_survival_rate
        num_to_select = int(round(pop_size * survival_rate))

        candidates = []
        for s in species_list:
            members = list(s.members.items())
            members.sort(reverse=True, key=lambda x: (x[1].fitness, x[0]))
            for gid, g in members:
                candidates.append((gid, g, s))

        elites = {}
        if self.reproduction_config.anji_elitism:
            for s in species_list:
                if len(s.members) >= self.reproduction_config.anji_elitism_min_species_size:
                    best_gid, best_g = max(
                        s.members.items(), key=lambda x: (x[1].fitness, x[0])
                    )
                    elites[best_gid] = best_g

        selected = dict(elites)
        if len(selected) > num_to_select:
            # Keep fittest elites only if elites exceed survival budget.
            elite_items = list(selected.items())
            elite_items.sort(reverse=True, key=lambda x: (x[1].fitness, x[0]))
            selected = dict(elite_items[:num_to_select])
            return selected

        for gid, g, _ in candidates:
            if len(selected) >= num_to_select:
                break
            selected.setdefault(gid, g)
        return selected

    @staticmethod
    def _normalize_species_allocations(raw_alloc, target_total):
        if not raw_alloc:
            return {}
        alloc = {k: int(round(v)) for k, v in raw_alloc.items()}
        diff = target_total - sum(alloc.values())
        keys = list(alloc.keys())
        i = 0
        while diff != 0 and keys:
            k = keys[i % len(keys)]
            if diff > 0:
                alloc[k] += 1
                diff -= 1
            else:
                if alloc[k] > 0:
                    alloc[k] -= 1
                    diff += 1
            i += 1
            if i > (target_total + 1) * max(1, len(keys)) and diff < 0:
                break
        return alloc

    def _allocate_by_species(self, species_list, offspring_count):
        if offspring_count <= 0 or not species_list:
            return {}
        fitness_by_sid = {s.key: max(0.0, self._species_fitness(s)) for s in species_list}
        total = sum(fitness_by_sid.values())
        if total <= 0.0:
            raw = {s.key: offspring_count / len(species_list) for s in species_list}
        else:
            raw = {s.key: (fitness_by_sid[s.key] / total) * offspring_count for s in species_list}
        return self._normalize_species_allocations(raw, offspring_count)

    def _clone_offspring(self, config, species_by_id, per_species_counts):
        offspring = []
        for sid, n in per_species_counts.items():
            if n <= 0:
                continue
            s = species_by_id[sid]
            parents = list(s.members.items())
            if not parents:
                continue
            for _ in range(n):
                pid, parent = random.choice(parents)
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent, parent, config.genome_config)
                self.ancestors[gid] = (pid, pid)
                offspring.append((gid, child))
        return offspring

    def _crossover_offspring(self, config, species_by_id, per_species_counts):
        offspring = []
        for sid, n in per_species_counts.items():
            if n <= 0:
                continue
            s = species_by_id[sid]
            parents = list(s.members.items())
            if not parents:
                continue
            for _ in range(n):
                if len(parents) == 1:
                    p1id, p1 = parents[0]
                    p2id, p2 = parents[0]
                else:
                    p1id, p1 = random.choice(parents)
                    p2id, p2 = random.choice(parents)
                    while p2id == p1id:
                        p2id, p2 = random.choice(parents)
                dominant, recessive = (p1, p2) if p1.fitness >= p2.fitness else (p2, p1)
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(dominant, recessive, config.genome_config)
                self.ancestors[gid] = (p1id, p2id)
                offspring.append((gid, child))
        return offspring

    def reproduce(self, config, species, pop_size, generation):
        # ANJI compatibility: persistent innovation mappings (no per-generation reset).
        config.genome_config.innovation_tracker = self.innovation_tracker

        remaining_species = []
        for sid, s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(sid, s)
            else:
                remaining_species.append(s)

        if not remaining_species:
            species.species = {}
            return {}

        survivors = self._select_survivors(remaining_species, pop_size)

        # Rebuild species membership to contain only survivors.
        species.species = {}
        survivor_ids = set(survivors.keys())
        for s in remaining_species:
            kept = {gid: g for gid, g in s.members.items() if gid in survivor_ids}
            if kept:
                s.members = kept
                species.species[s.key] = s

        species_after_selection = list(species.species.values())
        if not species_after_selection:
            # Fallback: keep best genome globally if selection emptied all species.
            best_gid, best_g = max(survivors.items(), key=lambda x: (x[1].fitness, x[0]))
            survivors = {best_gid: best_g}
            remaining_species[0].members = {best_gid: best_g}
            species.species = {remaining_species[0].key: remaining_species[0]}
            species_after_selection = list(species.species.values())

        offspring_needed = max(0, pop_size - len(survivors))
        survival_rate = self.reproduction_config.anji_survival_rate

        clone_slice = self.reproduction_config.anji_clone_slice
        crossover_slice = self.reproduction_config.anji_crossover_slice
        clone_slice = survival_rate if clone_slice == "auto" else float(clone_slice)
        crossover_slice = (1.0 - 2.0 * survival_rate) if crossover_slice == "auto" else float(crossover_slice)
        if clone_slice < 0.0 or crossover_slice < 0.0:
            raise RuntimeError("ANJI slice values must be non-negative.")

        target_clone = int(round(pop_size * clone_slice))
        target_cross = int(round(pop_size * crossover_slice))
        total_target = target_clone + target_cross
        if total_target <= 0:
            target_clone = offspring_needed
            target_cross = 0
        else:
            scale = offspring_needed / float(total_target)
            target_clone = int(round(target_clone * scale))
            target_cross = max(0, offspring_needed - target_clone)

        species_by_id = {s.key: s for s in species_after_selection}
        clone_alloc = self._allocate_by_species(species_after_selection, target_clone)
        cross_alloc = self._allocate_by_species(species_after_selection, target_cross)

        offspring = []
        offspring.extend(self._clone_offspring(config, species_by_id, clone_alloc))
        offspring.extend(self._crossover_offspring(config, species_by_id, cross_alloc))

        # Rounding reconciliation: adjust offspring count to exact target.
        while len(offspring) > offspring_needed:
            idx = random.randrange(len(offspring))
            offspring.pop(idx)
        while len(offspring) < offspring_needed and offspring:
            _, clonee = random.choice(offspring)
            gid = next(self.genome_indexer)
            clone_child = config.genome_type(gid)
            clone_child.configure_crossover(clonee, clonee, config.genome_config)
            self.ancestors[gid] = (clonee.key, clonee.key)
            offspring.append((gid, clone_child))

        # ANJI-style: mutate offspring, not survivors.
        for _, child in offspring:
            child.mutate(config.genome_config)

        new_population = dict(survivors)
        for gid, child in offspring:
            new_population[gid] = child
        return new_population
