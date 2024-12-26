import shlex
import subprocess as sb


def exc_cmd(cmd, **kwargs):
    """
    Execute a subprocess Popen and catch except.

    Import!! This method doesn't accept shell=True.

    :param cmd: The command to be execute.
    :param kwargs: All subprocess.Popen(..) kwargs except shell.
    :return: The stdout.
    :except excepts: SubprocessError, CalledProcessError

    :keyword stdin: Default=subprocess.PIPE
    :keyword stdout: Default=subprocess.PIPE
    :keyword stderr: Default=subprocess.PIPE
    :keyword encoding: Default=utf-8
    """
    stdout = kwargs.pop("stdout", sb.PIPE)
    stdin = kwargs.pop("stdin", sb.PIPE)
    stderr = kwargs.pop("stderr", sb.PIPE)
    encoding = kwargs.pop("encoding", "utf-8")

    cmd = shlex.split(cmd)
    with sb.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr, shell=False, encoding=encoding, **kwargs) as p:
        try:
            std_out, std_err = p.communicate(timeout=3600)
        except sb.TimeoutExpired:
            p.kill()
            std_out, std_err = p.communicate()  # try again if timeout fails.
        if p.returncode != 0:  # Bad error.
            raise sb.CalledProcessError(p.returncode, std_err)
        elif len(std_err) != 0:  # Some possible errors trowed by the running subprocess, but not critical.
            raise sb.SubprocessError(std_err)
        return std_out



def open_html_file(html_file_path):
    import webbrowser
    import os

    """
    Opens an HTML file in the default web browser.

    :param html_file_path: Path to the HTML file (e.g., 'path/to/index.html')
    """

    if not os.path.exists(html_file_path):
        print(f"Error: The file '{html_file_path}' does not exist.")
        return

    try:
        # Open the file in the default web browser
        webbrowser.open(f"file://{os.path.abspath(html_file_path)}")
        print(f"Opening: {html_file_path}")
    except Exception as e:
        print(f"Failed to open the file: {e}")

def open_url(url):
    import webbrowser
    import os
    """
    Opens a specified URL in the default web browser.

    :param url: The URL to open (e.g., 'https://isp.com')
    """
    try:
        webbrowser.open(url)
        print(f"Opening: {url}")
    except Exception as e:
        print(f"Failed to open the URL: {e}")


def exc_nll(cmd, **kwargs):
    """
    Execute a subprocess Popen and catch except.
    :param cmd: The command to execute.
    :param kwargs: All subprocess.Popen(..) kwargs except shell.
    :return: The stdout.
    :except excepts: SubprocessError, CalledProcessError
    :keyword stdin: Default=subprocess.PIPE
    :keyword stdout: Default=subprocess.PIPE
    :keyword stderr: Default=subprocess.PIPE
    :keyword encoding: Default=utf-8
    :keyword timeout: Default=3600
    """
    stdout = kwargs.pop("stdout", sb.PIPE)
    stdin = kwargs.pop("stdin", sb.PIPE)
    stderr = kwargs.pop("stderr", sb.PIPE)
    encoding = kwargs.pop("encoding", "utf-8")
    timeout = kwargs.pop("timeout", 3600)

    # cmd = shlex.split(cmd)
    with sb.Popen(cmd, stdin=stdin, stdout=stdout, stderr=stderr, encoding=encoding, **kwargs) as p:
        try:
            std_out, std_err = p.communicate(timeout=3600)
        except sb.TimeoutExpired:
            p.kill()
            std_out, std_err = p.communicate()  # try again if timeout fails.

        # Print the stdout and stderr to the terminal
        print(std_out)

        if p.returncode != 0:  # Bad error.
            raise sb.CalledProcessError(p.returncode, std_err)
        elif len(std_err) != 0:
            # Some possible errors trowed by the running subprocess, but not critical.
            print("stderr:", std_err)
            #raise sb.SubprocessError(std_err)
        return std_out
#
# if __name__ == "__main__":
#
#     command = ['/Users/robertocabiecesdiaz/Documents/SurfQuake/venv/lib/python3.9/site-packages/surfquakecore/binaries/mac_bin/NLL/scat2latlon', "1",
#                './', '/Users/robertocabiecesdiaz/Desktop/all_andorra/nll_all/loc/location.20211001.021028.grid0.loc']
#
#     additional_kwargs = {
#         "stdout": sb.PIPE,
#         "stderr": sb.PIPE,
#         "encoding": "utf-8",
#         "timeout": 10
#     }
#
#     result = exc_cmd(command, **additional_kwargs)
