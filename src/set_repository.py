"""Set the real group GitHub URL in report metadata, then recompile the PDF."""
from pathlib import Path
import argparse, json, re

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    args=parser.parse_args()
    url=args.url.rstrip('/')
    if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',url):
        parser.error('Use the real https://github.com/OWNER/REPOSITORY URL without query parameters.')
    root=Path(__file__).resolve().parents[1]
    path=root/'submission_config.json'
    config=json.loads(path.read_text(encoding='utf8'))
    config['github_url']=url
    path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    text='\\newcommand{\\repositorynotice}{The project repository is available at \\url{'+url+'}.}\n'
    (root/'report/submission_metadata.tex').write_text(text,encoding='utf8')
    print('Updated the real repository URL. Recompile report/main.tex and inspect the new PDF before submission.')

if __name__=='__main__':main()
