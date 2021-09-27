import collections
import redis
import numpy as np
from representations import StateAction
import random
import jericho
from jericho.template_action_generator import *
from jericho.defines import TemplateAction
import textworld

GraphInfo = collections.namedtuple('GraphInfo', 'objs, ob_rep, act_rep, graph_state, graph_state_rep, admissible_actions, admissible_actions_rep')

def load_vocab(env):
    vocab = {i+2: str(v) for i, v in enumerate(env.get_dictionary())}
    vocab[0] = ' '
    vocab[1] = '<s>'
    vocab_rev = {v: i for i, v in vocab.items()}
    return vocab, vocab_rev

def clean_obs(s):
    garbage_chars = ['*', '-', '!', '[', ']']
    for c in garbage_chars:
        s = s.replace(c, ' ')
    return s.strip()

class TemplateActionGeneratorJeri:
    '''
    Generates actions using the template-action-space.
    :param rom_bindings: Game-specific bindings from :meth:`jericho.FrotzEnv.bindings`.
    :type rom_bindings: Dictionary
    '''
    def __init__(self, rom_bindings):
        self.rom_bindings = rom_bindings
        grammar = rom_bindings['grammar'].split(';')
        max_word_length = rom_bindings['max_word_length']
        self.templates = self._preprocess_templates(grammar, max_word_length)
        # Enchanter and Spellbreaker only recognize abbreviated directions
        if rom_bindings['name'] in ['enchanter', 'spellbrkr', 'murdac']:
            for act in ['northeast','northwest','southeast','southwest']:
                self.templates.remove(act)
            self.templates.extend(['ne','nw','se','sw'])

    def _preprocess_templates(self, templates, max_word_length):
        '''
        Converts templates with multiple verbs and takes the first verb.
        '''
        out = []
        vb_usage_fn = lambda verb: verb_usage_count(verb, max_word_length)
        p = re.compile(r'\S+(/\S+)+')
        for template in templates:
            if not template:
                continue
            while True:
                match = p.search(template)
                if not match:
                    break
                verb = max(match.group().split('/'), key=vb_usage_fn)
                template = template[:match.start()] + verb + template[match.end():]
            ts = template.split()
            out.append(template)
        return out


    def generate_actions(self, objs):
        '''
        Given a list of objects present at the current location, returns
        a list of possible actions. This list represents all combinations
        of templates filled with the provided objects.
        :param objs: Candidate interactive objects present at the current location.
        :type objs: List of strings
        :returns: List of action-strings.
        :Example:
        >>> import jericho
        >>> env = jericho.FrotzEnv(rom_path)
        >>> interactive_objs = ['phone', 'keys', 'wallet']
        >>> env.act_gen.generate_actions(interactive_objs)
        ['wake', 'wake up', 'wash', ..., 'examine wallet', 'remove phone', 'taste keys']
        '''
        actions = []
        for template in self.templates:
            holes = template.count('OBJ')
            if holes <= 0:
                actions.append(template)
            elif holes == 1:
                actions.extend([template.replace('OBJ', obj) for obj in objs])
            elif holes == 2:
                for o1 in objs:
                    for o2 in objs:
                        if o1 != o2:
                            actions.append(template.replace('OBJ', o1, 1).replace('OBJ', o2, 1))
        return actions


    def generate_template_actions(self, objs, obj_ids):
        '''
        Given a list of objects and their corresponding vocab_ids, returns
        a list of possible TemplateActions. This list represents all combinations
        of templates filled with the provided objects.
        :param objs: Candidate interactive objects present at the current location.
        :type objs: List of strings
        :param obj_ids: List of ids corresponding to the tokens of each object.
        :type obj_ids: List of int
        :returns: List of :class:`jericho.defines.TemplateAction`.
        :Example:
        >>> import jericho
        >>> env = jericho.FrotzEnv(rom_path)
        >>> interactive_objs = ['phone', 'keys', 'wallet']
        >>> interactive_obj_ids = [718, 325, 64]
        >>> env.act_gen.generate_template_actions(interactive_objs, interactive_obj_ids)
        [
          TemplateAction(action='wake', template_id=0, obj_ids=[]),
          TemplateAction(action='wake up', template_id=1, obj_ids=[]),
          ...
          TemplateAction(action='turn phone on', template_id=55, obj_ids=[718]),
          TemplateAction(action='put wallet on keys', template_id=65, obj_ids=[64, 325])
         ]
        '''
        assert len(objs) == len(obj_ids)
        actions = []
        for template_idx, template in enumerate(self.templates):
            holes = template.count('OBJ')
            if holes <= 0:
                actions.append(defines.TemplateAction(template, template_idx, []))
            elif holes == 1:
                for noun, noun_id in zip(objs, obj_ids):
                    actions.append(
                        defines.TemplateAction(template.replace('OBJ', noun),
                                               template_idx, [noun_id]))
            elif holes == 2:
                for o1, o1_id in zip(objs, obj_ids):
                    for o2, o2_id in zip(objs, obj_ids):
                        if o1 != o2:
                            actions.append(
                                defines.TemplateAction(
                                    template.replace('OBJ', o1, 1).replace('OBJ', o2, 1),
                                    template_idx, [o1_id, o2_id]))
        return actions

class JeriWorld:
    def __init__(self, story_file, seed=None, style='jericho'):
        self._env = textworld.start(story_file)
        state = self._env.reset()
        self.tw_games = True
        self._seed = seed
        self.gp = None
        self.in7 = None
        self.bindings = None
        if state.command_templates is None:
            self.tw_games = False
            del self._env
            self._env = jericho.FrotzEnv(story_file, seed)
            self.bindings = self._env.bindings
            self._world_changed = self._env._world_changed
            self.act_gen = self._env.act_gen
        else:
            self.bindings = _load_bindings_from_tw(state, story_file, seed)
            self._world_changed = self._env._jericho._world_changed
            self.act_gen = TemplateActionGeneratorJeri(self.bindings)
            self.seed(seed)

    def __del__(self):
        del self._env
    
    def reset(self):
        if self.tw_games:
            state = self._env.reset()
            self.gp = GameProgression(state.game)
            self.in7 = Inform7Game(state.game)
            raw = re.split('-=.*?=-', state.raw)[1]
            return raw, {'moves':state.moves, 'score':state.score}
        return self._env.reset()
    
    def load(self, story_file, seed=None):
        self._env.load(self, story_file, seed=None)

    def step(self, action):
        if self.tw_games:
            old_score = self._env.state.score
            next_state = self._env.step(action)[0]
            s_action = re.sub(r'\s+', ' ', action.strip())
            try:
                idx = self.in7.gen_commands_from_actions(self.gp.valid_actions).index(s_action)
                self.gp.update(self.gp.valid_actions[idx])
            except:
                pass
            score = self._env.state.score
            reward = score - old_score
            self._world_changed = self._env._jericho._world_changed
            return next_state.raw, reward, (next_state.lost or next_state.won),\
              {'moves':next_state.moves, 'score':next_state.score}
        else:
            self._world_changed = self._env._world_changed
        return self._env.step(action)

    def bindings(self):
        return self.bindings

    def _emulator_halted(self):
        if self.tw_games:
            return self._env._env._emulator_halted()
        return self._env._emulator_halted()

    def game_over(self):
        if self.tw_games:
            self._env.state['lost']
        return self._env.game_over()

    def victory(self):
        if self.tw_games:
            self._env.state['won']
        return self._env.victory()

    def seed(self, seed=None):
        self._seed = seed
        return self._env.seed(seed)
    
    def close(self):
        self._env.close()

    def copy(self):
        return self._env.copy()

    def get_walkthrough(self):
        if self.tw_games:
            return self._env.state['extra.walkthrough']
        return self._env.get_walkthrough()

    def get_score(self):
        if self.tw_games:
            return self._env.state['score']
        return self._env.get_score()

    def get_dictionary(self):
        if self.tw_games:
            state = self._env.state
            return state.entities + state.verbs
        return self._env.get_dictionary()

    def get_state(self):
        if self.tw_games:
            return self._env._jericho.get_state()
        return self._env.get_state
    
    def set_state(self, state):
        if self.tw_games:
            self._env._jericho.set_state(state)
        else:
            self._env.get_state

    def get_valid_actions(self, use_object_tree=True, use_ctypes=True, use_parallel=True):
        if self.tw_games:
            return self.in7.gen_commands_from_actions(self.gp.valid_actions)
        return self._env.get_valid_actions(use_object_tree, use_ctypes, use_parallel)
    
    def _identify_interactive_objects(self, observation='', use_object_tree=False):
        """
        Identifies objects in the current location and inventory that are likely
        to be interactive.
        :param observation: (optional) narrative response to the last action, used to extract candidate objects.
        :type observation: string
        :param use_object_tree: Query the :doc:`object_tree` for names of surrounding objects.
        :type use_object_tree: boolean
        :returns: A list-of-lists containing the name(s) for each interactive object.
        :Example:
        >>> from jericho import *
        >>> env = FrotzEnv('zork1.z5')
        >>> obs, info = env.reset()
        'You are standing in an open field west of a white house with a boarded front door. There is a small mailbox here.'
        >>> env.identify_interactive_objects(obs)
        [['mailbox', 'small'], ['boarded', 'front', 'door'], ['white', 'house']]
        .. note:: Many objects may be referred to in a variety of ways, such as\
        Zork1's brass latern which may be referred to either as *brass* or *lantern*.\
        This method groups all such aliases together into a list for each object.
        """
        if self.tw_games:
            objs = set()
            state = self.get_state()

            if observation:
                # Extract objects from observation
                obs_objs = extract_objs(observation)
                obs_objs = [o + ('OBS',) for o in obs_objs]
                objs = objs.union(obs_objs)

            # Extract objects from location description
            self.set_state(state)
            look = clean(self.step('look')[0])
            look_objs = extract_objs(look)
            look_objs = [o + ('LOC',) for o in look_objs]
            objs = objs.union(look_objs)

            # Extract objects from inventory description
            self.set_state(state)
            inv = clean(self.step('inventory')[0])
            inv_objs = extract_objs(inv)
            inv_objs = [o + ('INV',) for o in inv_objs]
            objs = objs.union(inv_objs)
            self.set_state(state)

            # Filter out the objects that aren't in the dictionary
            dict_words = [w for w in self.get_dictionary()]
            max_word_length = max([len(w) for w in dict_words])
            to_remove = set()
            for obj in objs:
                if len(obj[0].split()) > 1:
                    continue
                if obj[0][:max_word_length] not in dict_words:
                    to_remove.add(obj)
            objs.difference_update(to_remove)
            objs_set = set()
            for obj in objs:
                if obj[0] not in objs_set:
                    objs_set.add(obj[0])
            return objs_set
        return self._env._identify_interactive_objects(observation=observation, use_object_tree=use_object_tree)

    def find_valid_actions(self):
        diff2acts = {}
        state = self.get_state()
        candidate_actions = self.get_valid_actions()
        for act in candidate_actions:
            self.set_state(state)
            self.step(act)
            diff = self._env._jericho._get_world_diff()
            if diff in diff2acts:
                if act not in diff2acts[diff]:
                    diff2acts[diff].append(act)
            else:
                diff2acts[diff] = [act]
        self.set_state(state)
        return diff2acts

    def find_valid_actions(self, possible_acts):
        admissible = []
        candidate_acts = self._env._filter_candidate_actions(possible_acts).values()
        true_actions = self._env.get_valid_actions()
        for temp_list in candidate_acts:
            for template in temp_list:
                if template.action in true_actions:
                    admissible.append(template)
        return admissible

    def _score_object_names(self, interactive_objs):
        """ Attempts to choose a sensible name for an object, typically a noun. """
        def score_fn(obj):
            score = -.01 * len(obj[0])
            if obj[1] == 'NOUN':
                score += 1
            if obj[1] == 'PROPN':
                score += .5
            if obj[1] == 'ADJ':
                score += 0
            if obj[2] == 'OBJTREE':
                score += .1
            return score
        best_names = []
        for desc, objs in interactive_objs.items():
            sorted_objs = sorted(objs, key=score_fn, reverse=True)
            best_names.append(sorted_objs[0][0])
        return best_names

    def get_world_state_hash(self):
        if self.tw_games:
            return None
        else:
            return self._env.get_world_state_hash()

class KGA2CEnv:
    '''

    KGA2C environment performs additional graph-based processing.

    '''
    def __init__(self, rom_path, seed, spm_model, tsv_file, step_limit=None, stuck_steps=10, gat=True):
        random.seed(seed)
        np.random.seed(seed)
        self.rom_path        = rom_path
        self.seed            = seed
        self.episode_steps   = 0
        self.stuck_steps     = 0
        self.valid_steps     = 0
        self.spm_model       = spm_model
        self.tsv_file        = tsv_file
        self.step_limit      = step_limit
        self.max_stuck_steps = stuck_steps
        self.gat             = gat
        self.env             = None
        self.conn_valid      = None
        self.conn_openie     = None
        self.vocab           = None
        self.vocab_rev       = None
        self.state_rep       = None

    def create(self):
        ''' Create the Jericho environment and connect to redis. '''
        self.env = JeriWorld(self.rom_path, self.seed)
        self.bindings = self.env.bindings
        self.act_gen = self.env.act_gen
        self.max_word_len = self.bindings['max_word_length']
        self.vocab, self.vocab_rev = load_vocab(self.env)
        self.conn_valid = redis.Redis(host='localhost', port=6379, db=0)
        self.conn_openie = redis.Redis(host='localhost', port=6379, db=1)


    def _get_admissible_actions(self, objs):
        ''' Queries Redis for a list of admissible actions from the current state. '''
        obj_ids = [self.vocab_rev[o[:self.max_word_len]] for o in objs]
        world_state_hash = self.env.get_world_state_hash()
        admissible = self.conn_valid.get(world_state_hash)
        if admissible is None:
            possible_acts = self.act_gen.generate_template_actions(objs, obj_ids)
            admissible = self.env.find_valid_actions(possible_acts)
            redis_valid_value = '/'.join([str(a) for a in admissible])
            self.conn_valid.set(world_state_hash, redis_valid_value)
        else:
            try:
                admissible = [eval(a.strip()) for a in admissible.decode('cp1252').split('/')]
            except Exception as e:
                print("Exception: {}. Admissible: {}".format(e, admissible))
        return admissible


    def _build_graph_rep(self, action, ob_r):
        ''' Returns various graph-based representations of the current state. '''
        #objs = [o[0] for o in self.env._identify_interactive_objects(ob_r)]
        objs = []
        for inter_objs in self.env._identify_interactive_objects().values():
            for obj in inter_objs:
                objs.append(obj[0])
        admissible_actions = self._get_admissible_actions(objs)
        admissible_actions_rep = [self.state_rep.get_action_rep_drqa(a.action) \
                                  for a in admissible_actions] \
                                      if admissible_actions else [[0] * 20]
        try: # Gather additional information about the new state
            save_state = self.env.get_state()
            ob_l = self.env.step('look')[0]
            self.env.set_state(save_state)
            ob_i = self.env.step('inventory')[0]
            self.env.set_state(save_state)
        except RuntimeError:
            print('RuntimeError: {}, Done: {}, Info: {}'.format(clean_obs(ob_r), done, info))
            ob_l = ob_i = ''
        ob_rep = self.state_rep.get_obs_rep(ob_l, ob_i, ob_r, action)
        cleaned_obs = clean_obs(ob_l + ' ' + ob_r)
        openie_cache = self.conn_openie.get(cleaned_obs)
        if openie_cache is None:
            rules, tocache = self.state_rep.step(cleaned_obs, ob_i, objs, action, cache=None, gat=self.gat)
            self.conn_openie.set(cleaned_obs, str(tocache))
        else:
            openie_cache = eval(openie_cache.decode('cp1252'))
            rules, _ = self.state_rep.step(cleaned_obs, ob_i, objs, action, cache=openie_cache, gat=self.gat)
        graph_state = self.state_rep.graph_state
        graph_state_rep = self.state_rep.graph_state_rep
        action_rep = self.state_rep.get_action_rep_drqa(action)
        return GraphInfo(objs, ob_rep, action_rep, graph_state, graph_state_rep,\
                         admissible_actions, admissible_actions_rep)
                         


    def step(self, action):
        self.episode_steps += 1
        obs, reward, done, info = self.env.step(action)
        info['valid'] = self.env._world_changed() or done
        info['steps'] = self.episode_steps
        if info['valid']:
            self.valid_steps += 1
            self.stuck_steps = 0
        else:
            self.stuck_steps += 1
        if (self.step_limit and self.valid_steps >= self.step_limit) \
           or self.stuck_steps > self.max_stuck_steps:
            done = True
        if done:
            graph_info = GraphInfo(objs=['all'],
                                   ob_rep=self.state_rep.get_obs_rep(obs, obs, obs, action),
                                   act_rep=self.state_rep.get_action_rep_drqa(action),
                                   graph_state=self.state_rep.graph_state,
                                   graph_state_rep=self.state_rep.graph_state_rep,
                                   admissible_actions=[],
                                   admissible_actions_rep=[])
        else:
            graph_info = self._build_graph_rep(action, obs)
        return obs, reward, done, info, graph_info


    def reset(self):
        self.state_rep = StateAction(self.spm_model, self.vocab, self.vocab_rev,
                                     self.tsv_file, self.max_word_len)
        self.stuck_steps = 0
        self.valid_steps = 0
        self.episode_steps = 0
        obs, info = self.env.reset()
        info['valid'] = False
        info['steps'] = 0
        graph_info = self._build_graph_rep('look', obs)
        return obs, info, graph_info


    def close(self):
        self.env.close()
