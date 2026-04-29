import os

import neat


def test_anji_mode_type_overrides():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "test_configuration_anji")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    assert config.algorithm_mode == "anji"
    assert config.genome_type is neat.AnjiGenome
    assert config.reproduction_type is neat.AnjiReproduction
    assert config.species_set_type is neat.AnjiSpeciesSet
    assert config.stagnation_type is neat.AnjiNoStagnation


def test_anji_mode_population_smoke():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "test_configuration_anji")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    pop = neat.Population(config, seed=7)

    def eval_genomes(genomes, cfg):
        del cfg
        for gid, genome in genomes:
            genome.fitness = float(gid % 5)

    winner = pop.run(eval_genomes, 2)
    assert winner is not None
    assert isinstance(pop.reproduction.innovation_tracker, neat.AnjiInnovationTracker)


def _build_prune_test_genome(config):
    gcfg = config.genome_config
    if gcfg.innovation_tracker is None:
        gcfg.innovation_tracker = neat.AnjiInnovationTracker()
    genome = neat.AnjiGenome(123)
    genome.configure_new(gcfg)

    # Remove auto-initialized wiring to keep topology deterministic for this test.
    genome.connections.clear()

    # Keep one valid hidden path: input -> 10 -> output
    hidden_keep = 10
    genome.nodes[hidden_keep] = genome.create_node(gcfg, hidden_keep)
    genome.add_connection(gcfg, gcfg.input_keys[0], hidden_keep, 1.0, True)
    genome.add_connection(gcfg, hidden_keep, gcfg.output_keys[0], 1.0, True)

    # Hidden node with no incoming path from inputs (should be pruned).
    hidden_no_input = 11
    genome.nodes[hidden_no_input] = genome.create_node(gcfg, hidden_no_input)
    genome.add_connection(gcfg, hidden_no_input, gcfg.output_keys[0], 1.0, True)

    # Hidden node with no path to outputs (should be pruned).
    hidden_no_output = 12
    genome.nodes[hidden_no_output] = genome.create_node(gcfg, hidden_no_output)
    genome.add_connection(gcfg, gcfg.input_keys[1], hidden_no_output, 1.0, True)

    # Hidden node with only self-loop (should be pruned).
    hidden_self_loop = 13
    genome.nodes[hidden_self_loop] = genome.create_node(gcfg, hidden_self_loop)
    genome.add_connection(gcfg, hidden_self_loop, hidden_self_loop, 1.0, True)

    return genome


def test_anji_prune_removes_stranded_hidden_nodes():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "test_configuration_anji")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    genome = _build_prune_test_genome(config)
    gcfg = config.genome_config
    genome._prune_anji_stranded_alleles(gcfg)

    assert 10 in genome.nodes
    assert 11 not in genome.nodes
    assert 12 not in genome.nodes
    assert 13 not in genome.nodes

    remaining = set(genome.connections.keys())
    assert (gcfg.input_keys[0], 10) in remaining
    assert (10, gcfg.output_keys[0]) in remaining
    assert (11, gcfg.output_keys[0]) not in remaining
    assert (gcfg.input_keys[1], 12) not in remaining
    assert (13, 13) not in remaining


def test_anji_prune_runs_via_mutate():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "test_configuration_anji")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    genome = _build_prune_test_genome(config)
    gcfg = config.genome_config

    # Force mutate() to execute prune step deterministically.
    gcfg.conn_add_prob = 0.0
    gcfg.node_add_prob = 0.0
    gcfg.anji_remove_connection_rate = 0.0
    gcfg.anji_prune = True
    gcfg.anji_prune_rate = 1.0
    gcfg.anji_mutate_nodes = False

    genome.mutate(gcfg)
    assert 10 in genome.nodes
    assert 11 not in genome.nodes
    assert 12 not in genome.nodes
    assert 13 not in genome.nodes
