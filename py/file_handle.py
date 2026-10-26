import os
import json
import csv
from dataclasses import fields, asdict
from cluster_diagnostics import Cluster_Diagnostics

class File_Handle:

    @staticmethod
    def load_json_file(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            return json.load(json_file)

    @staticmethod
    def save_xyzv_snapshot(xyzv_file_path, particle_positions, particle_velocities):
        with open(xyzv_file_path, "a", encoding="utf-8") as file:
            file.write(f"{len(particle_positions)}\n")
            file.write("Properties=species:S:1:pos:R:3:vel:R:3\n")

            for position, velocity in zip(particle_positions, particle_velocities):
                file.write(
                    f"H "
                    f"{position[0]:.16e} {position[1]:.16e} {position[2]:.16e} "
                    f"{velocity[0]:.16e} {velocity[1]:.16e} {velocity[2]:.16e}\n"
                )

    @staticmethod
    def open_json_snapshot(json_file_path):
        json_file_descriptor = open(json_file_path, "w", encoding="utf-8")
        json_file_descriptor.write('{"snapshots": [\n')
        return json_file_descriptor


    @staticmethod
    def save_json_snapshot(json_file_descriptor, particle_positions, particle_velocities):

        # we gotta write a comma if it's not the first snapshot
        if json_file_descriptor.tell() > len('{"snapshots": [\n'):
            json_file_descriptor.write(",\n")

        snapshot = {
            "number_of_particles": len(particle_positions),
            "particle_positions": particle_positions.tolist(),
            "particle_velocities": particle_velocities.tolist()
        }

        json.dump(snapshot, json_file_descriptor)
        json_file_descriptor.flush()

    @staticmethod
    def close_json_snapshot(json_file_descriptor):
        json_file_descriptor.write("\n]}\n")
        json_file_descriptor.close()


    @staticmethod
    def save_tsv_snapshot(tsv_file_path, cluster_diagnostics: Cluster_Diagnostics = None):

        fieldnames = [field.name for field in fields(Cluster_Diagnostics)]
        file_exists = os.path.exists(tsv_file_path)

        with open(tsv_file_path, "a", newline="") as tsv_file:
            writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, delimiter="\t")

            if not file_exists or os.path.getsize(tsv_file_path) == 0:
                writer.writeheader()

            writer.writerow(asdict(cluster_diagnostics))
