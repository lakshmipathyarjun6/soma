if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bathing-Dataset-MANO-MOSH')

    parser.add_argument('--data-dir', required=True, type=str, help='The path to the top-level data directory')
    parser.add_argument('--captures', required=True, type=str, help='The name of the captures type to run')
    parser.add_argument('--session', required=True, type=str, help='The name of the session to run')
    parser.add_argument('--task', required=False, default='mosh', type=str, help='Task to run. Can be "mosh" or "render", defaults to mosh')
    parser.add_argument('--hand', required=True, type=str, help='The name of the hand to use. Can be "left" or "right", defaults to right')

    args = parser.parse_args()

    bathing_work_base_dir = args.data_dir
    captures = args.captures
    session = args.session
    task = args.task
    hand = args.hand

    fix_mano_mosh(bathing_work_base_dir, captures, session, task, hand)
