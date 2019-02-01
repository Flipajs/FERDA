from core.project.project import Project


def project_to_open_format(dir_src, dir_dst):
    project = Project(dir_src)
    project.save_new(dir_dst)


if __name__ == '__main__':
    import fire
    fire.Fire(project_to_open_format)
