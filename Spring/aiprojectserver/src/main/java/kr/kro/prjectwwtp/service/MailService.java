package kr.kro.prjectwwtp.service;


import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.amazonaws.services.simpleemail.AmazonSimpleEmailService;
import com.amazonaws.services.simpleemail.model.Body;
import com.amazonaws.services.simpleemail.model.Content;
import com.amazonaws.services.simpleemail.model.Destination;
import com.amazonaws.services.simpleemail.model.Message;
import com.amazonaws.services.simpleemail.model.SendEmailRequest;
import com.amazonaws.services.simpleemail.model.SendEmailResult;

import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MailService {
	private final AmazonSimpleEmailService amazonSimpleEmailService;
	
	@Value("${aws.region}")
	private String region;
	@Value("${aws.ses.send-mail-from}")
	private String sendMailFrom;

	public void sendEmail(String toAddress, String subjectText, String bodyText) {
		Destination destination = new Destination().withToAddresses(toAddress);
		
		Content subject = new Content().withCharset("UTF-8").withData(subjectText);
		Content body = new Content().withCharset("UTF-8").withData(bodyText);
		
		Message message = new Message().withSubject(subject)
				.withBody(new Body().withHtml(body));
		
		SendEmailRequest  request = new SendEmailRequest()
			.withSource(sendMailFrom)
			.withDestination(destination)
			.withMessage(message);
		
		SendEmailResult result = amazonSimpleEmailService.sendEmail(request);
		System.out.println("Email send response: " + result);
	}
}
