import os
import random
import argparse
from multiprocessing import Pool

import numpy as np
from PIL import Image

import neat
from common import eval_mono_image, eval_gray_image, eval_color_image

width, height = 16, 16
full_scale = 16


def evaluate_lowres(genome, config, scheme):
    if scheme == 'gray':
        return eval_gray_image(genome, config, width, height)
    elif scheme == 'color':
        return eval_color_image(genome, config, width, height)
    elif scheme == 'mono':
        return eval_mono_image(genome, config, width, height)

    raise Exception(f'Unexpected scheme: {scheme!r}')


class NoveltyEvaluator:
    def __init__(self, num_workers, scheme):
        self.num_workers = num_workers
        self.scheme = scheme
        self.pool = Pool(num_workers)
        self.archive = []
        self.out_index = 1

    def image_from_array(self, image):
        if self.scheme == 'color':
            return Image.fromarray(image, mode="RGB")

        return Image.fromarray(image, mode="L")

    def evaluate(self, genomes, config):
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(evaluate_lowres, (genome, config, self.scheme)))

        new_archive_entries = []
        for (genome_id, genome), j in zip(genomes, jobs):
            image = np.clip(np.array(j.get()), 0, 255).astype(np.uint8)
            float_image = image.astype(np.float32) / 255.0

            genome.fitness = (width * height) ** 0.5
            for a in self.archive:
                adist = np.linalg.norm(float_image.ravel() - a.ravel())
                genome.fitness = min(genome.fitness, adist)

            if random.random() < 0.02:
                new_archive_entries.append(float_image)
                # im = self.image_from_array(image)
                # im.save("novelty-{0:06d}.png".format(self.out_index))

                if self.scheme == 'gray':
                    image = eval_gray_image(genome, config, full_scale * width, full_scale * height)
                elif self.scheme == 'color':
                    image = eval_color_image(genome, config, full_scale * width, full_scale * height)
                elif self.scheme == 'mono':
                    image = eval_mono_image(genome, config, full_scale * width, full_scale * height)
                else:
                    raise Exception(f'Unexpected scheme: {self.scheme!r}')

                im = np.clip(np.array(image), 0, 255).astype(np.uint8)
                im = self.image_from_array(im)
                im.save(f'novelty-{self.out_index:06d}.png')

                self.out_index += 1

        self.archive.extend(new_archive_entries)
        print(f'{len(self.archive)} archive entries')


def run(config_filename='novelty_config'):
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    if os.path.isabs(config_filename):
        config_path = config_filename
    else:
        config_path = os.path.join(local_dir, config_filename)
    config_basename = os.path.basename(config_path)
    run_dir = os.path.join(local_dir, f'exp-{config_basename}')
    os.makedirs(run_dir, exist_ok=True)
    # Note that we provide the custom stagnation class to the Config constructor.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    snapshot_interval = max(1, int(getattr(config, "snapshot_interval", 100)))

    previous_cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        ne = NoveltyEvaluator(4, 'color')
        if ne.scheme == 'color':
            config.output_nodes = 3
        else:
            config.output_nodes = 1

        pop = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        print(f"Run directory: {run_dir}")
        while 1:
            pop.run(ne.evaluate, 1)

            winner = stats.best_genome()
            if ne.scheme == 'gray':
                image = eval_gray_image(winner, config, full_scale * width, full_scale * height)
            elif ne.scheme == 'color':
                image = eval_color_image(winner, config, full_scale * width, full_scale * height)
            elif ne.scheme == 'mono':
                image = eval_mono_image(winner, config, full_scale * width, full_scale * height)
            else:
                raise Exception(f'Unexpected scheme: {ne.scheme!r}')

            im = np.clip(np.array(image), 0, 255).astype(np.uint8)
            im = ne.image_from_array(im)
            im.save(f'winning-novelty-{pop.generation:06d}.png')

            if ne.scheme == 'gray':
                image = eval_gray_image(winner, config, width, height)
            elif ne.scheme == 'color':
                image = eval_color_image(winner, config, width, height)
            elif ne.scheme == 'mono':
                image = eval_mono_image(winner, config, width, height)
            else:
                raise Exception(f'Unexpected scheme: {ne.scheme!r}')

            float_image = np.array(image, dtype=np.float32) / 255.0
            ne.archive.append(float_image)

            if pop.generation % snapshot_interval == 0:
                snapshot_dir = f"snapshot-{pop.generation:05d}"
                os.makedirs(snapshot_dir, exist_ok=True)
                with open(os.path.join(snapshot_dir, "winner.bin"), "wb") as f:
                    import pickle
                    pickle.dump(winner, f, 2)
                im.save(os.path.join(snapshot_dir, "winner.png"))
                print("\n" + "=" * 72)
                print(
                    f" SNAPSHOT SAVED: generation {pop.generation:05d} -> "
                    f"{os.path.abspath(snapshot_dir)}"
                )
                print("=" * 72 + "\n")
    finally:
        os.chdir(previous_cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run novelty picture2d evolution with a chosen config file.'
    )
    parser.add_argument(
        'config_filename',
        nargs='?',
        default='novelty_config',
        help='Config file name relative to this script, or an absolute path.',
    )
    args = parser.parse_args()
    run(args.config_filename)
