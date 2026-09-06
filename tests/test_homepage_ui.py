"""UI-only tests: no Streamlit dependency, database, test asset, or worker."""
import base64
from contextlib import nullcontext
from pathlib import Path
import unittest

from medcl import homepage_ui as ui

class StubColumn:
    def __init__(self, st): self.st = st
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def button(self, *args, **kwargs): return self.st.button(*args, **kwargs)
    def html(self, *args, **kwargs): return self.st.html(*args, **kwargs)

class StubStreamlit:
    def __init__(self): self.htmls=[]; self.buttons={}; self.captions=[]
    def container(self, **kwargs): return nullcontext()
    def columns(self, spec, **kwargs): return [StubColumn(self) for _ in range(spec if isinstance(spec,int) else len(spec))]
    def html(self, value): self.htmls.append(value)
    def caption(self, value): self.captions.append(value)
    def button(self, label, **kwargs):
        key=kwargs['key']
        if key in self.buttons: raise AssertionError(f'Duplicate widget key: {key}')
        self.buttons[key]=(label,kwargs); return False

class HomepageUITests(unittest.TestCase):
    def setUp(self):
        self.st=StubStreamlit(); self.events=[]
        self.jobs=[]; self.benchmarks=[]
    def render(self, **kwargs):
        ui.render_homepage(self.st,benchmarks=self.benchmarks,jobs=self.jobs,
            readiness=lambda b: (b.get('ready',False),''),
            kinds={'segmentation':'分割','classification':'分类','registration':'配准'},
            statuses={'completed':'已完成','failed':'失败','queued':'排队中','running':'评测中'},
            on_start=lambda:self.events.append(('start',)),on_records=lambda:self.events.append(('records',)),
            on_task=lambda k:self.events.append(('task',k)),on_job=lambda j:self.events.append(('job',j)),**kwargs)
    def click(self,key):
        _,k=self.st.buttons[key]; k['on_click'](*k.get('args',()))
    def test_assets_are_bundled_webp(self):
        for name in ui._ASSETS:
            data=base64.b64decode(ui.asset_uri(name).split(',',1)[1])
            self.assertEqual(data[:4],b'RIFF'); self.assertEqual(data[8:12],b'WEBP')
    def test_asset_allowlist(self):
        for path in ('../../settings.local.json','/etc/passwd','real-patient.png'):
            with self.assertRaises(ValueError): ui.asset_uri(path)
    def test_all_icons_are_local_images(self):
        for name in ui._ICON_PATHS:
            markup=ui.icon(name)
            self.assertIn('<img ',markup); self.assertNotIn('<svg',markup)
            svg=base64.b64decode(ui.icon_uri(name).split(',',1)[1]).decode()
            self.assertIn('viewBox="0 0 24 24"',svg)
            self.assertNotIn('<script',svg)
    def test_icon_input_validation(self):
        with self.assertRaises(ValueError): ui.icon_uri('script')
        with self.assertRaises(ValueError): ui.icon_uri('brain','red" onclick="bad')
        with self.assertRaises(ValueError): ui.icon('brain',size=9999)
    def test_stylesheet_resolves_all_tokens(self):
        css=ui.stylesheet()
        self.assertNotIn('__MEDICAL_',css); self.assertNotIn('__ICON_',css)
        self.assertIn('data:image/webp;base64,',css)
        self.assertNotIn('https://',css); self.assertNotIn('http://',css)
    def test_showcase_is_explicit_illustration(self):
        html=ui.hero_visual_html()
        self.assertIn('功能示意 · 非实验结果 · 非真实病例',html)
        self.assertNotIn('真实标注',html)
        self.assertEqual(html.count('<figure>'),4)
        self.assertEqual(html.count('is-missing'),6)
    def test_empty_homepage_does_not_invent_records(self):
        self.render()
        html=''.join(self.st.htmls)
        self.assertIn('还没有评测记录',html)
        self.assertNotIn('2024-',html)
        self.assertFalse(any(k.startswith('home-job-') for k in self.st.buttons))
    def test_callbacks_are_wired_to_existing_actions(self):
        self.render()
        for key in ('medcl-home-start','medcl-home-records','medcl-home-all-records',
                    'home-enter-segmentation','home-enter-classification','home-enter-registration'):
            self.click(key)
        self.assertEqual(self.events,[('start',),('records',),('records',),('task','segmentation'),
                                       ('task','classification'),('task','registration')])
    def test_recent_selection_keeps_exact_job_id(self):
        self.jobs=[{'id':'job-7','created_at':'2026-09-06T02:00:00+00:00','status':'completed',
                    'config':{'method':'基线','benchmark':{'title':'真实协议','kind':'segmentation'}}}]
        self.render(); self.click('home-job-job-7')
        self.assertEqual(self.events,[('job','job-7')])
    def test_untrusted_titles_are_escaped(self):
        self.jobs=[{'id':'job-x','created_at':'2026-09-06','status':'completed',
            'config':{'method':'<script>alert(1)</script>','benchmark':{'title':'\"><img src=x onerror=evil()>','kind':'classification'}}}]
        self.render(); html=''.join(self.st.htmls)
        self.assertNotIn('<script>',html); self.assertNotIn('<img src=x',html)
        self.assertIn('&lt;script&gt;',html)
    def test_only_ready_real_protocols_are_counted(self):
        self.benchmarks=[{'kind':'segmentation','synthetic':False,'ready':True},
            {'kind':'segmentation','synthetic':True,'ready':True},
            {'kind':'registration','synthetic':False,'ready':False}]
        self.render(); html=''.join(self.st.htmls)
        self.assertIn('1 个可用真实协议',html)
        self.assertNotIn('2 个可用真实协议',html)
    def test_latest_three_records_are_selected(self):
        self.jobs=[{'id':str(i),'created_at':f'2026-09-0{i}T00:00:00','status':'queued',
                    'config':{'method':'工程示例','benchmark':{'title':'合成验收','kind':'classification'}}} for i in range(1,5)]
        self.render()
        self.assertEqual([k for k in self.st.buttons if k.startswith('home-job-')],['home-job-4','home-job-3','home-job-2'])
    def test_invalid_overview_values_rejected(self):
        for value in (True,-1,1.0):
            with self.assertRaises(ValueError): ui.overview_html([('数值',value,'bars','blue')])
    def test_development_records_disclosed(self):
        self.render(show_demos=True)
        self.assertTrue(any('包含当前可见的合成验收记录' in c for c in self.st.captions))
    def test_header_preserves_required_selection_guard(self):
        calls=[]
        def control(*args): calls.append(args); return '首页'
        nav=['首页','任务中心','评测记录','方法比较']
        self.assertEqual(ui.render_header(self.st,nav,control),'首页')
        self.assertEqual(calls,[('主导航',nav,'nav','collapsed')])
        self.assertIn('data:image/svg+xml;base64,',''.join(self.st.htmls))
    def test_rendering_does_not_mutate_input_data(self):
        import copy
        self.benchmarks=[{'kind':'segmentation','synthetic':False,'ready':False}]
        original=copy.deepcopy((self.jobs,self.benchmarks)); self.render()
        self.assertEqual((self.jobs,self.benchmarks),original)

if __name__=='__main__': unittest.main()
