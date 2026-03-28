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
