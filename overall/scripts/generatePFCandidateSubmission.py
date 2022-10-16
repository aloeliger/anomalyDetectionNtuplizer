#A utility for generating a script that will submit our even and odd event number ntuplization

import argparse
import datetime
import os
import sys

def main(args):
    
    finalSubmissionScript = open(args.submissionName+'_finalSubmission.sh', 'w')
    finalSubmissionScript.write('#Submission generated with command: '+' '.join(sys.argv))
    finalSubmissionScript.write('\n')

    
    for evenOrOdd in ['Even','Odd']:
        dagLocation = args.submissionPath+'/'+args.submissionName+'/'+evenOrOdd+'/dags'
        os.system('mkdir -p '+dagLocation)

        submitDir = args.submissionPath+'/'+args.submissionName+'/'+evenOrOdd+'/submit'
        
        configLocation = os.environ["CMSSW_BASE"]+f'/src/anomalyDetectionNtuplizer/overall/python/random{evenOrOdd}EventNtuplization_cff.py'

        command = [
            'farmoutAnalysisJobs --vsize-limit 8000 --memory-requirements=8000',
            '--infer-cmssw-path',
            f'"--submit-dir={submitDir}"',
            f'"--output-dag-file={dagLocation}/dag"',
            f'"--output-dir=/hdfs/store/user/aloeliger/{args.submissionName}/{evenOrOdd}/"',
            f'--input-files-per-job={args.filesPerJob}',
            f'--input-file-list={args.inputFile}',
            '--assume-input-files-exist',
            '--input-dir=/',
            f'{args.submissionName}-{evenOrOdd}',
            configLocation,
            "'inputFiles=$inputFileNames'",
            "'outputFile=$outputFileName'",
        ]
        finalFarmoutCommand = ' '.join(command)+'\n'
        finalSubmissionScript.write('#'+evenOrOdd+' submission command\n')
        finalSubmissionScript.write(finalFarmoutCommand)
        finalSubmissionScript.write('\n')
    finalSubmissionScript.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Create submission file to ntuplize the pf candidates')
    parser.add_argument('--submissionPath',
                        nargs='?',
                        default='/nfs_scratch/'+os.environ['USER']+'/')
    parser.add_argument('--submissionName',
                        nargs='?',
                        default=datetime.datetime.now().strftime('%d%b%Y_%H%M_PFCandSubmission'),
                        help='Name to store scratch submissions under, and to store result tuples in the store.')
    parser.add_argument('--inputFile',
                        nargs='?',
                        required=True,
                        help='Text file listing all the events to be ntuplized')
    parser.add_argument('--filesPerJob', 
                        nargs='?',
                        type=int, 
                        default=1, 
                        help='Number of input files each job should handle')

    args = parser.parse_args()

    main(args)
