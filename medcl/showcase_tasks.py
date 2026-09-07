"""Fixed, public task identities; private gallery paths are derived from these IDs."""

SCENARIOS = {"domain": "Domain-CL · 域增量", "class": "Class-CL · 类别增量", "task": "Task-CL · 任务增量"}
CLASS_LAYOUTS = {"pathmnist": ((0, 1), (2, 3), (4, 5), (6, 7, 8)),
                 "skin": ((0, 1), (2, 3), (4, 5)),
                 "hyperkvasir": tuple((i, i + 1) for i in range(0, 20, 2))}
SEGMENTATION_TASKS = {
    "domain": (("BIDMC", "Domain_Prostate/BIDMC.h5", 0), ("HK", "Domain_Prostate/HK.h5", 0),
               ("ISBI", "Domain_Prostate/ISBI.h5", 0), ("UCL", "Domain_Prostate/UCL.h5", 0),
               ("ISBI-1.5", "Domain_Prostate/ISBI_1.5.h5", 0), ("I2CVB", "Domain_Prostate/I2CVB.h5", 0)),
    "class": (("MYO / LV / LA", "MMWHS/myo_lv_la.h5", 0), ("RA / RV", "MMWHS/ra_rv.h5", 3),
              ("AO / PA", "MMWHS/ao_pa.h5", 5)),
    "task": (("左心房", "Task_incre/UtahI.h5", 0), ("前列腺", "Task_incre/UCL.h5", 0),
             ("肝脏", "Task_incre/Lits.h5", 0), ("脑肿瘤", "Task_incre/brain.h5", 0)),
}
REGISTRATION_TASKS = (("OASIS · 脑 MRI", "oasis"), ("CTCT · 腹部 CT", "ctct"),
                      ("NLST · 肺部 CT", "nlst"), ("MRCT · 腹部 MR–CT", "mrct"))


def task_specs(kind, *, scenario="domain", dataset="pathmnist"):
    if kind == "classification":
        return [{"id": f"T{i}", "name": "类别 " + " / ".join(map(str, classes)), "classes": classes,
                 "file": f"classification-{dataset}-T{i}"} for i, classes in enumerate(CLASS_LAYOUTS[dataset], 1)]
    if kind == "segmentation":
        return [{"id": f"T{i}", "name": name, "source": source, "shift": shift,
                 "file": f"segmentation-{scenario}-T{i}"}
                for i, (name, source, shift) in enumerate(SEGMENTATION_TASKS[scenario], 1)]
    if kind == "registration":
        return [{"id": f"T{i}", "name": name, "source": source, "file": f"registration-task-T{i}"}
                for i, (name, source) in enumerate(REGISTRATION_TASKS, 1)]
    raise ValueError("Unknown showcase task kind")
